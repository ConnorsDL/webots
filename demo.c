/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description:   Autonomous vehicle controller example extended with
 *                wheel-speed decoding and IMU-based slip compensation.
 */

#include <webots/accelerometer.h>
#include <webots/camera.h>
#include <webots/device.h>
#include <webots/display.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/lidar.h>
#include <webots/position_sensor.h>
#include <webots/robot.h>
#include <webots/vehicle/driver.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// to be used as array indices
enum { X, Y, Z };

#define TIME_STEP 50
#define UNKNOWN 99999.99
#define MAX_WHEEL_SENSORS 4

// Line following PID
#define KP 0.25
#define KI 0.006
#define KD 2

// Longitudinal speed control (fused speed)
#define SPEED_KP 0.45
#define SPEED_KI 0.08
#define SPEED_INTEGRAL_LIMIT 60.0
#define MAX_CRUISE_SPEED_KMH 80.0
#define MIN_CRUISE_SPEED_KMH 0.0
#define CRUISE_SLEW_KMH 2.0

// Slip handling
#define SLIP_RATIO_THRESHOLD 0.20
#define SLIP_ABS_THRESHOLD 1.5      // m/s
#define SLIP_BRAKE_GAIN 0.30
#define MAX_SLIP_BRAKE 0.35

// Wheel decoding (generic passenger-car wheel radius)
#define WHEEL_RADIUS_M 0.32

bool PID_need_reset = false;

// Size of the yellow line angle filter
#define FILTER_SIZE 3

// enable various 'features'
bool enable_collision_avoidance = false;
bool enable_display = false;
bool has_gps = false;
bool has_camera = false;

// camera
WbDeviceTag camera;
int camera_width = -1;
int camera_height = -1;
double camera_fov = -1.0;

// SICK laser
WbDeviceTag sick;
int sick_width = -1;
double sick_fov = -1.0;

// speedometer
WbDeviceTag display;
WbImageRef speedometer_image = NULL;

// GPS
WbDeviceTag gps;
double gps_coords[3] = {0.0, 0.0, 0.0};
double gps_speed = 0.0;

// IMU-related tags
WbDeviceTag imu = 0;
WbDeviceTag gyro = 0;
WbDeviceTag accelerometer = 0;

// Wheel speed decoder inputs
WbDeviceTag wheel_sensors[MAX_WHEEL_SENSORS];
double prev_wheel_pos[MAX_WHEEL_SENSORS];
bool prev_wheel_valid[MAX_WHEEL_SENSORS];
int wheel_sensor_count = 0;

// fused odometry/control state
double enc_speed_ms = 0.0;
double imu_speed_ms = 0.0;
double fused_speed_ms = 0.0;
double target_speed_kmh = 50.0;
double speed = 0.0;
double speed_control_integral = 0.0;
double steering_angle = 0.0;
int manual_steering = 0;
bool autodrive = true;

static double clamp(double v, double lo, double hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

void print_help() {
  printf("You can drive this car!\n");
  printf("Select the 3D window and then use the cursor keys to:\n");
  printf("[LEFT]/[RIGHT] - steer\n");
  printf("[UP]/[DOWN] - set target speed\n");
  printf("[A] - toggle back to auto-drive\n");
}

void set_autodrive(bool onoff) {
  if (autodrive == onoff)
    return;
  autodrive = onoff;
  switch (autodrive) {
    case false:
      printf("switching to manual drive...\n");
      printf("hit [A] to return to auto-drive.\n");
      break;
    case true:
      if (has_camera)
        printf("switching to auto-drive...\n");
      else
        printf("impossible to switch auto-drive on without camera...\n");
      break;
  }
}

void set_speed(double kmh) {
  target_speed_kmh = clamp(kmh, 0.0, 250.0);
  printf("target speed set to %.1f km/h\n", target_speed_kmh);
}

// positive: turn right, negative: turn left
void set_steering_angle(double wheel_angle) {
  if (wheel_angle - steering_angle > 0.1)
    wheel_angle = steering_angle + 0.1;
  if (wheel_angle - steering_angle < -0.1)
    wheel_angle = steering_angle - 0.1;
  steering_angle = wheel_angle;
  if (wheel_angle > 0.5)
    wheel_angle = 0.5;
  else if (wheel_angle < -0.5)
    wheel_angle = -0.5;
  wbu_driver_set_steering_angle(wheel_angle);
}

void change_manual_steer_angle(int inc) {
  set_autodrive(false);

  double new_manual_steering = manual_steering + inc;
  if (new_manual_steering <= 25.0 && new_manual_steering >= -25.0) {
    manual_steering = new_manual_steering;
    set_steering_angle(manual_steering * 0.02);
  }

  if (manual_steering == 0)
    printf("going straight\n");
  else
    printf("turning %.2f rad (%s)\n", steering_angle, steering_angle < 0 ? "left" : "right");
}

void check_keyboard() {
  int key = wb_keyboard_get_key();
  switch (key) {
    case WB_KEYBOARD_UP:
      set_speed(target_speed_kmh + 5.0);
      break;
    case WB_KEYBOARD_DOWN:
      set_speed(target_speed_kmh - 5.0);
      break;
    case WB_KEYBOARD_RIGHT:
      change_manual_steer_angle(+1);
      break;
    case WB_KEYBOARD_LEFT:
      change_manual_steer_angle(-1);
      break;
    case 'A':
      set_autodrive(true);
      break;
  }
}

int color_diff(const unsigned char a[3], const unsigned char b[3]) {
  int i, diff = 0;
  for (i = 0; i < 3; i++) {
    int d = a[i] - b[i];
    diff += d > 0 ? d : -d;
  }
  return diff;
}

double process_camera_image(const unsigned char *image) {
  int num_pixels = camera_height * camera_width;
  const unsigned char REF[3] = {95, 187, 203};
  int sumx = 0;
  int pixel_count = 0;

  const unsigned char *pixel = image;
  int x;
  for (x = 0; x < num_pixels; x++, pixel += 4) {
    if (color_diff(pixel, REF) < 30) {
      sumx += x % camera_width;
      pixel_count++;
    }
  }

  if (pixel_count == 0)
    return UNKNOWN;

  return ((double)sumx / pixel_count / camera_width - 0.5) * camera_fov;
}

double filter_angle(double new_value) {
  static bool first_call = true;
  static double old_value[FILTER_SIZE];
  int i;

  if (first_call || new_value == UNKNOWN) {
    first_call = false;
    for (i = 0; i < FILTER_SIZE; ++i)
      old_value[i] = 0.0;
  } else {
    for (i = 0; i < FILTER_SIZE - 1; ++i)
      old_value[i] = old_value[i + 1];
  }

  if (new_value == UNKNOWN)
    return UNKNOWN;

  old_value[FILTER_SIZE - 1] = new_value;
  double sum = 0.0;
  for (i = 0; i < FILTER_SIZE; ++i)
    sum += old_value[i];
  return sum / FILTER_SIZE;
}

double process_sick_data(const float *sick_data, double *obstacle_dist) {
  const int HALF_AREA = 20;
  int sumx = 0;
  int collision_count = 0;
  int x;
  *obstacle_dist = 0.0;
  for (x = sick_width / 2 - HALF_AREA; x < sick_width / 2 + HALF_AREA; x++) {
    float range = sick_data[x];
    if (range < 20.0) {
      sumx += x;
      collision_count++;
      *obstacle_dist += range;
    }
  }

  if (collision_count == 0)
    return UNKNOWN;

  *obstacle_dist = *obstacle_dist / collision_count;
  return ((double)sumx / collision_count / sick_width - 0.5) * sick_fov;
}

void update_display() {
  const double NEEDLE_LENGTH = 50.0;

  wb_display_image_paste(display, speedometer_image, 0, 0, false);

  double current_speed = wbu_driver_get_current_speed();
  if (isnan(current_speed))
    current_speed = 0.0;
  double alpha = current_speed / 260.0 * 3.72 - 0.27;
  int x = -NEEDLE_LENGTH * cos(alpha);
  int y = -NEEDLE_LENGTH * sin(alpha);
  wb_display_draw_line(display, 100, 95, 100 + x, 95 + y);

  char txt[128];
  sprintf(txt, "GPS:%.1fkm/h ENC:%.1fkm/h", gps_speed, enc_speed_ms * 3.6);
  wb_display_draw_text(display, txt, 10, 130);
  sprintf(txt, "IMU:%.1fkm/h FUSED:%.1fkm/h", imu_speed_ms * 3.6, fused_speed_ms * 3.6);
  wb_display_draw_text(display, txt, 10, 140);
}

void compute_gps_speed() {
  const double *coords = wb_gps_get_values(gps);
  const double speed_ms = wb_gps_get_speed(gps);
  gps_speed = speed_ms * 3.6;
  memcpy(gps_coords, coords, sizeof(gps_coords));
}

void detect_and_enable_wheel_sensors() {
  wheel_sensor_count = 0;
  const int device_count = wb_robot_get_number_of_devices();
  for (int i = 0; i < device_count && wheel_sensor_count < MAX_WHEEL_SENSORS; ++i) {
    WbDeviceTag device = wb_robot_get_device_by_index(i);
    const char *name = wb_device_get_name(device);
    if (name == NULL)
      continue;
    // Typical names in Webots car models include rear/front wheel sensors and wheel encoders.
    if (strstr(name, "wheel sensor") || strstr(name, "Wheel Sensor") || strstr(name, "encoder")) {
      wheel_sensors[wheel_sensor_count] = device;
      wb_position_sensor_enable(device, TIME_STEP);
      prev_wheel_valid[wheel_sensor_count] = false;
      printf("enabled wheel sensor: %s\n", name);
      wheel_sensor_count++;
    }
  }
  if (wheel_sensor_count == 0)
    printf("warning: no wheel position sensors found; controller will fallback to IMU/GPS speed only.\n");
}

void detect_and_enable_imu_sensors() {
  const int device_count = wb_robot_get_number_of_devices();
  for (int i = 0; i < device_count; ++i) {
    WbDeviceTag device = wb_robot_get_device_by_index(i);
    const char *name = wb_device_get_name(device);
    if (!name)
      continue;

    if (!imu && (strcmp(name, "inertial unit") == 0 || strstr(name, "IMU") || strstr(name, "imu"))) {
      imu = device;
      wb_inertial_unit_enable(imu, TIME_STEP);
    } else if (!gyro && (strcmp(name, "gyro") == 0 || strstr(name, "gyro"))) {
      gyro = device;
      wb_gyro_enable(gyro, TIME_STEP);
    } else if (!accelerometer &&
               (strcmp(name, "accelerometer") == 0 || strstr(name, "accelerometer") || strstr(name, "acc"))) {
      accelerometer = device;
      wb_accelerometer_enable(accelerometer, TIME_STEP);
    }
  }

  if (imu)
    printf("IMU orientation source enabled.\n");
  if (gyro)
    printf("gyro enabled.\n");
  if (accelerometer)
    printf("accelerometer enabled.\n");
  if (!accelerometer)
    printf("warning: no accelerometer found; slip correction degrades to encoder-only speed estimate.\n");
}

void update_fused_speed(double dt) {
  // Wheel speed decoder: convert wheel angle increments to linear velocity.
  if (wheel_sensor_count > 0) {
    double omega_abs_sum = 0.0;
    int valid_count = 0;
    for (int i = 0; i < wheel_sensor_count; ++i) {
      double wheel_pos = wb_position_sensor_get_value(wheel_sensors[i]);
      if (isnan(wheel_pos))
        continue;
      if (prev_wheel_valid[i]) {
        const double dtheta = wheel_pos - prev_wheel_pos[i];
        const double omega = dtheta / dt;
        omega_abs_sum += fabs(omega);
        valid_count++;
      }
      prev_wheel_pos[i] = wheel_pos;
      prev_wheel_valid[i] = true;
    }

    if (valid_count > 0)
      enc_speed_ms = (omega_abs_sum / valid_count) * WHEEL_RADIUS_M;
  }

  // IMU forward acceleration integration with tilt compensation.
  double ax_forward = 0.0;
  if (accelerometer) {
    const double *acc = wb_accelerometer_get_values(accelerometer);
    if (acc) {
      double pitch = 0.0;
      if (imu) {
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
        if (rpy)
          pitch = rpy[Y];
      }
      // Approximate longitudinal acceleration by removing gravity projection on X.
      const double gravity_projection = 9.81 * sin(pitch);
      ax_forward = acc[X] - gravity_projection;
    }
  }

  imu_speed_ms += ax_forward * dt;
  if (has_gps)
    imu_speed_ms = 0.98 * imu_speed_ms + 0.02 * (gps_speed / 3.6);

  // Slip-aware fusion
  double slip = enc_speed_ms - imu_speed_ms;
  const double slip_ratio = (fused_speed_ms > 0.5) ? fabs(slip) / (fused_speed_ms + 1e-6) : 0.0;
  const bool slip_detected = fabs(slip) > SLIP_ABS_THRESHOLD && slip_ratio > SLIP_RATIO_THRESHOLD;

  if (slip_detected)
    fused_speed_ms = 0.3 * enc_speed_ms + 0.7 * imu_speed_ms;
  else
    fused_speed_ms = 0.7 * enc_speed_ms + 0.3 * imu_speed_ms;

  if (fused_speed_ms < 0.0)
    fused_speed_ms = 0.0;
}

void update_longitudinal_control(double dt) {
  const double target_ms = target_speed_kmh / 3.6;
  const double error = target_ms - fused_speed_ms;

  speed_control_integral = clamp(speed_control_integral + error * dt, -SPEED_INTEGRAL_LIMIT, SPEED_INTEGRAL_LIMIT);

  double cruise_delta = SPEED_KP * error + SPEED_KI * speed_control_integral;
  cruise_delta = clamp(cruise_delta, -CRUISE_SLEW_KMH, CRUISE_SLEW_KMH);

  speed = clamp(speed + cruise_delta, MIN_CRUISE_SPEED_KMH, MAX_CRUISE_SPEED_KMH);
  wbu_driver_set_cruising_speed(speed);

  // Brake support in case of clear overspeed/slip.
  const double slip = enc_speed_ms - imu_speed_ms;
  const double slip_ratio = (fused_speed_ms > 0.5) ? fabs(slip) / (fused_speed_ms + 1e-6) : 0.0;
  const bool slip_detected = fabs(slip) > SLIP_ABS_THRESHOLD && slip_ratio > SLIP_RATIO_THRESHOLD;

  double brake = 0.0;
  if (error < -0.8)
    brake = clamp((-error) * 0.12, 0.0, 0.5);
  if (slip_detected && slip > 0.0) {
    double slip_brake = clamp((fabs(slip) - SLIP_ABS_THRESHOLD) * SLIP_BRAKE_GAIN, 0.0, MAX_SLIP_BRAKE);
    if (slip_brake > brake)
      brake = slip_brake;
  }
  wbu_driver_set_brake_intensity(brake);
}

double applyPID(double yellow_line_angle) {
  static double oldValue = 0.0;
  static double integral = 0.0;

  if (PID_need_reset) {
    oldValue = yellow_line_angle;
    integral = 0.0;
    PID_need_reset = false;
  }

  if (signbit(yellow_line_angle) != signbit(oldValue))
    integral = 0.0;

  double diff = yellow_line_angle - oldValue;
  if (integral < 30 && integral > -30)
    integral += yellow_line_angle;

  oldValue = yellow_line_angle;
  return KP * yellow_line_angle + KI * integral + KD * diff;
}

int main(int argc, char **argv) {
  wbu_driver_init();

  for (int j = 0; j < wb_robot_get_number_of_devices(); ++j) {
    WbDeviceTag device = wb_robot_get_device_by_index(j);
    const char *name = wb_device_get_name(device);
    if (strcmp(name, "Sick LMS 291") == 0)
      enable_collision_avoidance = true;
    else if (strcmp(name, "display") == 0)
      enable_display = true;
    else if (strcmp(name, "gps") == 0)
      has_gps = true;
    else if (strcmp(name, "camera") == 0)
      has_camera = true;
  }

  if (has_camera) {
    camera = wb_robot_get_device("camera");
    wb_camera_enable(camera, TIME_STEP);
    camera_width = wb_camera_get_width(camera);
    camera_height = wb_camera_get_height(camera);
    camera_fov = wb_camera_get_fov(camera);
  }

  if (enable_collision_avoidance) {
    sick = wb_robot_get_device("Sick LMS 291");
    wb_lidar_enable(sick, TIME_STEP);
    sick_width = wb_lidar_get_horizontal_resolution(sick);
    sick_fov = wb_lidar_get_fov(sick);
  }

  if (has_gps) {
    gps = wb_robot_get_device("gps");
    wb_gps_enable(gps, TIME_STEP);
  }

  detect_and_enable_imu_sensors();
  detect_and_enable_wheel_sensors();

  if (enable_display) {
    display = wb_robot_get_device("display");
    speedometer_image = wb_display_image_load(display, "speedometer.png");
  }

  set_speed(50.0);
  speed = target_speed_kmh;
  wbu_driver_set_cruising_speed(speed);

  wbu_driver_set_hazard_flashers(true);
  wbu_driver_set_dipped_beams(true);
  wbu_driver_set_antifog_lights(true);
  wbu_driver_set_wiper_mode(SLOW);

  print_help();
  wb_keyboard_enable(TIME_STEP);

  while (wbu_driver_step() != -1) {
    check_keyboard();

    static int i = 0;
    if (i % (int)(TIME_STEP / wb_robot_get_basic_time_step()) == 0) {
      const double dt = (double)TIME_STEP / 1000.0;

      const unsigned char *camera_image = NULL;
      const float *sick_data = NULL;
      if (has_camera)
        camera_image = wb_camera_get_image(camera);
      if (enable_collision_avoidance)
        sick_data = wb_lidar_get_range_image(sick);

      if (has_gps)
        compute_gps_speed();

      update_fused_speed(dt);
      update_longitudinal_control(dt);

      if (autodrive && has_camera) {
        double yellow_line_angle = filter_angle(process_camera_image(camera_image));
        double obstacle_dist;
        double obstacle_angle;
        if (enable_collision_avoidance)
          obstacle_angle = process_sick_data(sick_data, &obstacle_dist);
        else {
          obstacle_angle = UNKNOWN;
          obstacle_dist = 0;
        }

        if (enable_collision_avoidance && obstacle_angle != UNKNOWN) {
          double obstacle_steering = steering_angle;
          if (obstacle_angle > 0.0 && obstacle_angle < 0.4)
            obstacle_steering = steering_angle + (obstacle_angle - 0.25) / obstacle_dist;
          else if (obstacle_angle > -0.4)
            obstacle_steering = steering_angle + (obstacle_angle + 0.25) / obstacle_dist;
          double steer = steering_angle;
          if (yellow_line_angle != UNKNOWN) {
            const double line_following_steering = applyPID(yellow_line_angle);
            if (obstacle_steering > 0 && line_following_steering > 0)
              steer = obstacle_steering > line_following_steering ? obstacle_steering : line_following_steering;
            else if (obstacle_steering < 0 && line_following_steering < 0)
              steer = obstacle_steering < line_following_steering ? obstacle_steering : line_following_steering;
          } else
            PID_need_reset = true;
          set_steering_angle(steer);
        } else if (yellow_line_angle != UNKNOWN) {
          set_steering_angle(applyPID(yellow_line_angle));
        } else {
          wbu_driver_set_brake_intensity(0.4);
          PID_need_reset = true;
        }
      }

      if (enable_display)
        update_display();
    }

    ++i;
  }

  wbu_driver_cleanup();
  return 0;
}

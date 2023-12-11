import numpy as np
import matplotlib.pyplot as plt

class TrapezoidalProfile:
    def __init__(self, max_velocity, time_interval):
        self.max_velocity = max_velocity
        self.time_interval = time_interval

    def generate_profile(self, target_position):
        current_position = 0.0
        current_velocity = 0.0
        time = 0.0

        profile = {'time': [], 'position': [], 'velocity': []}

        while current_position < target_position:
            acceleration = self.calculate_acceleration(current_position, target_position, current_velocity)

            if current_velocity < self.max_velocity:
                current_velocity += acceleration * self.time_interval
                if current_velocity > self.max_velocity:
                    current_velocity = self.max_velocity

            current_position += current_velocity * self.time_interval

            time += self.time_interval

            profile['time'].append(time)
            profile['position'].append(current_position)
            profile['velocity'].append(current_velocity)

        return profile

    def calculate_acceleration(self, current_position, target_position, current_velocity):
        distance_to_go = target_position - current_position

        time_to_target = max(0, distance_to_go / self.max_velocity)

        acceleration = 2 * (distance_to_go - current_velocity * time_to_target) / time_to_target**2

        return acceleration

def main():
    max_velocity = 1.0  # Set the maximum velocity of the actuators
    time_interval = 0.01  # Set the time interval for simulation

    trapezoidal_profile = TrapezoidalProfile(max_velocity, time_interval)

    target_position = 5.0  # Set the target position for the actuators

    # Generate trapezoidal motion profiles for each actuator
    profile_actuator_1 = trapezoidal_profile.generate_profile(1)
    profile_actuator_2 = trapezoidal_profile.generate_profile(2)
    profile_actuator_3 = trapezoidal_profile.generate_profile(3)

    # Plot the motion profiles
    plt.plot(profile_actuator_1['time'], profile_actuator_1['position'], label='Actuator 1')
    plt.plot(profile_actuator_2['time'], profile_actuator_2['position'], label='Actuator 2')
    plt.plot(profile_actuator_3['time'], profile_actuator_3['position'], label='Actuator 3')

    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Trapezoidal Motion Profile for Robot Arm Actuators')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

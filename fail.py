import numpy as np
import matplotlib.pyplot as plt

N = 18
d = 0.5
snapshots = 5000
SNR_dB = 30

start_angle = -10
end_angle = 40
desired_angle_path = np.linspace(start_angle, end_angle, snapshots)

interferer_angle = -50

def steering_vector(angle_deg, N, d):
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi * d
    n = np.arange(N)
    return np.exp(1j * k * n * np.sin(angle_rad))

def generate_qpsk_signal(num_symbols):
    bits = np.random.randint(0, 4, num_symbols)
    return (1/np.sqrt(2)) * np.exp(1j * (np.pi/4 + np.pi/2 * bits))

def generate_rayleigh_fading(num_symbols, num_paths=1):
    return (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)

a_i = steering_vector(interferer_angle, N, d)

desired_signal = generate_qpsk_signal(snapshots)
interferer_signal = generate_qpsk_signal(snapshots) * 5

fading_desired = generate_rayleigh_fading(snapshots)
fading_interferer = generate_rayleigh_fading(snapshots)

snr = 10**(SNR_dB / 10)
noise_power = 1 / snr
noise = np.sqrt(noise_power/2) * (np.random.randn(N, snapshots) + 1j * np.random.randn(N, snapshots))

weights = np.zeros(N, dtype=complex)
weights[0] = 1

mu = 0.0001 # Slightly reduced learning rate for more stability
output_signal = np.zeros(snapshots, dtype=complex)

beam_peak_history = []
angle_grid = np.linspace(-90, 90, 360)

for t in range(snapshots):
    current_desired_angle = desired_angle_path[t]
    a_d = steering_vector(current_desired_angle, N, d)

    received_signal_t = (a_d * desired_signal[t] * fading_desired[t] +
                         a_i * interferer_signal[t] * fading_interferer[t])

    x_t = received_signal_t.reshape(-1, 1) + noise[:, t].reshape(-1, 1)

    y_t = np.conj(weights).T @ x_t
    output_signal[t] = y_t.item()

    e_t = (np.abs(y_t)**2 - 1) * y_t
    weights = weights - mu * np.conj(e_t) * x_t.flatten()

    # FIX: Normalize weights at each step to prevent them from growing uncontrollably (exploding).
    # This is a key step for ensuring algorithm stability, which resolves the overflow error.
    weights = weights / np.linalg.norm(weights)

    if t % 50 == 0:
        beam_pattern_responses = []
        for angle in angle_grid:
            s_vec = steering_vector(angle, N, d)
            beam_pattern_responses.append(np.abs(np.conj(weights).T @ s_vec))
        beam_pattern = np.array(beam_pattern_responses)
        beam_peak_history.append(angle_grid[np.argmax(beam_pattern)])

print("CMA algorithm finished.")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Advanced Adaptive Beamforming with Moving User and Fading', fontsize=16)

ax1 = plt.subplot(2, 2, 1, projection='polar')
final_beam_pattern_responses = []
for angle in angle_grid:
    s_vec = steering_vector(angle, N, d)
    final_beam_pattern_responses.append(np.abs(np.conj(weights).T @ s_vec))
final_beam_pattern = np.array(final_beam_pattern_responses)

ax1.plot(np.deg2rad(angle_grid), final_beam_pattern)
ax1.plot(np.deg2rad(desired_angle_path[-1]), np.max(final_beam_pattern), 'go', markersize=12, label='User Final Position')
ax1.plot(np.deg2rad(interferer_angle), np.max(final_beam_pattern), 'ro', markersize=12, label='Interferer')
ax1.set_title('Final Beam Pattern')
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
ax2.scatter(np.real(output_signal[-500:]), np.imag(output_signal[-500:]), alpha=0.5)
ax2.set_title('Output Signal Constellation (Final 500 Symbols)')
ax2.set_xlabel('In-Phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.grid(True)
ax2.set_aspect('equal')

ax3 = plt.subplot(2, 1, 2)
time_axis = np.arange(0, snapshots, 50)
ax3.plot(time_axis, desired_angle_path[::50], 'g-', lw=2, label='True User Path')
ax3.plot(time_axis, beam_peak_history, 'b--o', markersize=4, label='Beamformer Pointing Direction')
ax3.set_title('Angle Tracking Performance')
ax3.set_xlabel('Time Snapshot')
ax3.set_ylabel('Angle (Degrees)')
ax3.grid(True)
ax3.legend()
ax3.set_ylim(-90, 90)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#include <array>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>

extern "C"
{

void sample(const char* electrons_filename, const char* ions_filename,
  double peak_density, double sigma_r, double sigma_pr, double mot_z_offset,
  double laser_phi_x, double laser_phi_y, double laser_x, double laser_y,
  double laser_z, double laser_width, unsigned seed, bool ions_on);

}

[[noreturn]] static void new_signalhandler(int signum)
{
  std::fprintf(stderr, "interrupt recieved (%d), exiting\n", signum);
  std::exit(signum);
}

class SignalHandlerHelper {
public:
  SignalHandlerHelper();
  ~SignalHandlerHelper();
private:
  void (*m_previous_signalhandler)(int);
};

SignalHandlerHelper::SignalHandlerHelper()
{
  m_previous_signalhandler = std::signal(SIGINT, new_signalhandler);
}

SignalHandlerHelper::~SignalHandlerHelper()
{
  std::signal(SIGINT, m_previous_signalhandler);
}

static constexpr double m_electron = 9.109383701528e-31;
static constexpr double m_ion = 1.4192261e-25;
static constexpr double elementary_charge = 1.602176634e-19;
static constexpr double c_light = 299792458;
static constexpr double m_electron_c = m_electron * c_light;
static constexpr double m_ion_c = m_ion * c_light;

void sample(const char* electrons_filename, const char* ions_filename,
  double peak_density, double sigma_r, double sigma_pr, double mot_z_offset,
  double laser_phi_x, double laser_phi_y, double laser_x, double laser_y,
  double laser_z, double laser_width, unsigned seed, bool ions_on)
{
  // Initialize Signal Handling
  SignalHandlerHelper signal_handler_helper{};

  // Initialize Random Numbers
  std::random_device random_device{};
  std::mt19937 random_generator{seed == 0 ? random_device() : seed};

  // Open File
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(electrons_filename, std::ofstream::out | std::ofstream::trunc);
  file << std::setprecision(32) << std::scientific;

  // Initialize Distributions
  std::normal_distribution<> xy_distribution{0.0, sigma_r};
  std::normal_distribution<> z_distribution{mot_z_offset, sigma_r};
  std::normal_distribution<> transverse_momentum_distribution{0.0, sigma_pr};

  // Compute Total Particles
  std::size_t total_particles = static_cast<std::size_t>(std::round(
    peak_density * std::pow(2 * M_PI, 1.5)
    * sigma_r * sigma_r * sigma_r));
  std::cout << "total mot particles: " << total_particles << std::endl;

  // Compute Laser Angle Vector
  double tx = std::tan(laser_phi_x);
  double ty = std::tan(laser_phi_y);
  double nx = tx / std::sqrt(1 + tx * tx + ty * ty);
  double ny = ty / std::sqrt(1 + tx * tx + ty * ty);
  double nz = 1 / std::sqrt(1 + tx * tx + ty * ty);

  // Compute Laser Rotation Matrix
  double theta = std::atan(std::sqrt(tx * tx + ty * ty));
  double beta = std::atan2(ty, tx);
  double c_theta = std::cos(theta);
  double s_theta = std::sin(theta);
  double c_beta = std::cos(beta);
  double s_beta = std::sin(beta);
  double rotation_xx = c_theta * c_beta;
  double rotation_xy = -s_beta;
  double rotation_xz = s_theta * c_beta;
  double rotation_yx = c_theta * s_beta;
  double rotation_yy = c_beta;
  double rotation_yz = s_theta * s_beta;
  double rotation_zx = -s_theta;
  double rotation_zy = 0.0;
  double rotation_zz = c_theta;

  // Shift Laser Center
  laser_z += mot_z_offset;

  // Function that Samples Positions
  auto sample_position = [&]() -> std::optional<std::array<double, 3>> {
    double z = z_distribution(random_generator);
    if (z > 0)
      return std::nullopt;
    double x = xy_distribution(random_generator);
    double y = xy_distribution(random_generator);
    double a_minus_p_x = laser_x - x;
    double a_minus_p_y = laser_y - y;
    double a_minus_p_z = laser_z - z;
    double dot_product = a_minus_p_x * nx + a_minus_p_y * ny + a_minus_p_z * nz;
    double distance_x = a_minus_p_x - dot_product * nx;
    double distance_y = a_minus_p_y - dot_product * ny;
    double distance_z = a_minus_p_z - dot_product * nz;
    double distance = std::sqrt(distance_x * distance_x + distance_y * distance_y
      + distance_z * distance_z);
    if (distance > laser_width)
      return std::nullopt;
    return std::array<double, 3>({x, y, z});
  };

  // Function that Samples Momenta
  auto sample_momentum = [&]() -> std::array<double, 3> {
    double momentum_x = transverse_momentum_distribution(random_generator);
    double momentum_y = transverse_momentum_distribution(random_generator);
    double momentum_z = 0.0;
    double rotated_momentum_x = rotation_xx * momentum_x + rotation_xy * momentum_y + rotation_xz * momentum_z;
    double rotated_momentum_y = rotation_yx * momentum_x + rotation_yy * momentum_y + rotation_yz * momentum_z;
    double rotated_momentum_z = rotation_zx * momentum_x + rotation_zy * momentum_y + rotation_zz * momentum_z;
    return std::array<double, 3>({rotated_momentum_x, rotated_momentum_y, rotated_momentum_z});
  };

  // Write Columns
  file << "x y z GBx GBy GBz m q nmacro\n";

  // Sample Electrons
  std::size_t n_sampled_electrons = 0;
  for (std::size_t i = 0; i != total_particles; ++i)
  {
    auto position = sample_position();
    if (!position)
      continue;
    ++n_sampled_electrons;
    auto momentum = sample_momentum();
    file << (*position)[0] << ' ' << (*position)[1] << ' ' << (*position)[2]
         << ' ' << (momentum[0] / m_electron_c) << ' '
         << (momentum[1] / m_electron_c) << ' ' << (momentum[2] / m_electron_c)
         << ' ' << m_electron << ' ' << -elementary_charge << ' ' << 1.0 << '\n';
  }
  std::cout << n_sampled_electrons << " electrons sampled" << std::endl;

  file.close();

  if (ions_on) {

    file.open(ions_filename, std::ofstream::out | std::ofstream::trunc);

    file << "x y z GBx GBy GBz m q nmacro\n";

    // Sample Ions
    std::size_t n_sampled_ions = 0;
    std::size_t j = 0;
    while (n_sampled_ions != n_sampled_electrons)
    {
      ++j;
      auto position = sample_position();
      if (!position)
        continue;
      ++n_sampled_ions;
      auto momentum = sample_momentum();
      file << (*position)[0] << ' ' << (*position)[1] << ' ' << (*position)[2]
           << ' ' << (momentum[0] / m_ion_c) << ' '
           << (momentum[1] / m_ion_c) << ' ' << (momentum[2] / m_ion_c)
           << ' ' << m_ion << ' ' << elementary_charge << ' ' << 1.0 << '\n';
    }
    std::cout << "sampled " << n_sampled_ions << " ions (" << (j - n_sampled_ions) << " rejected)" << std::endl;

    file.close();
  }
}

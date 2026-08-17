#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <cstdint>

namespace py = pybind11;

struct Coord {
    int x, y, z;
    Coord(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

py::array_t<int> propagate_3d(py::array_t<uint8_t> input, py::tuple rect) {
    auto buf = input.request();
    uint8_t* data = static_cast<uint8_t*>(buf.ptr);
    
    if (buf.ndim != 3)
        throw std::runtime_error("Input must be a 3D array");
    int z_dim = buf.shape[0];
    int y_dim = buf.shape[1];
    int x_dim = buf.shape[2];
    
    if (rect.size() != 6)
        throw std::runtime_error("rect must contain 6 integers");
    int xmin = rect[0].cast<int>();
    int xmax = rect[1].cast<int>();
    int ymin = rect[2].cast<int>();
    int ymax = rect[3].cast<int>();
    int zmin = rect[4].cast<int>();
    int zmax = rect[5].cast<int>();
    
    // Создаём массив с правильной формой (z, y, x)
    py::array_t<int> result({z_dim, y_dim, x_dim});
    auto res_buf = result.request();
    int* lat = static_cast<int*>(res_buf.ptr);
    
    std::fill(lat, lat + buf.size, 0);
    
    std::queue<Coord> q;
    
    for (int z = zmin; z <= zmax; ++z) {
        if (z < 0 || z >= z_dim) continue;
        for (int y = ymin; y <= ymax; ++y) {
            if (y < 0 || y >= y_dim) continue;
            for (int x = xmin; x <= xmax; ++x) {
                if (x < 0 || x >= x_dim) continue;
                int idx = (z * y_dim + y) * x_dim + x;
                if (data[idx] == 255) {
                    lat[idx] = 1;
                    q.emplace(x, y, z);
                }
            }
        }
    }
    
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};
    
    while (!q.empty()) {
        Coord cur = q.front();
        q.pop();
        int cur_idx = (cur.z * y_dim + cur.y) * x_dim + cur.x;
        int step = lat[cur_idx];
        
        for (int d = 0; d < 6; ++d) {
            int nx = cur.x + dx[d];
            int ny = cur.y + dy[d];
            int nz = cur.z + dz[d];
            if (nx >= 0 && nx < x_dim && ny >= 0 && ny < y_dim && nz >= 0 && nz < z_dim) {
                int nidx = (nz * y_dim + ny) * x_dim + nx;
                if (data[nidx] == 255 && lat[nidx] == 0) {
                    lat[nidx] = step + 1;
                    q.emplace(nx, ny, nz);
                }
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(bfs_propagate, m) {
    m.doc() = "BFS propagation (3D) implemented in C++";
    m.def("propagate_3d", &propagate_3d, "Perform BFS propagation on a 3D binary image",
          py::arg("input"), py::arg("rect"));
}
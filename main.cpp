
#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>

#include <GL/glext.h>
#include "glu3.h"

#include <math.h>
#include <stdlib.h>
#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>

#include "glu_complement.h"
#include "threadpool.hpp"

template <class Int>
class range_class {
 public:
  range_class(Int e) : m_end(e) {}
  range_class begin() const { return 0; }
  range_class end() const { return m_end; }
  void operator++() { m_end++; }
  Int operator*() const { return m_end; }
  bool operator!=(const range_class &e) const { return m_end < e.m_end; }

 private:
  Int m_end;
};

template <class Int>
const range_class<Int> range(Int i) {
  return range_class<Int>(i);
}

struct V3f {
  std::array<float, 3> p = {};
  void operator+=(const V3f &rhs) {
    for (auto i : range(p.size())) p[i] += rhs.p[i];
  }
  void operator-=(const V3f &rhs) {
    for (auto i : range(p.size())) p[i] -= rhs.p[i];
  }
  void operator*=(const float rhs) {
    for (auto i : range(p.size())) p[i] *= rhs;
  }
  V3f operator-(const V3f &rhs) const {
    V3f r(*this);
    r -= rhs;
    return r;
  }
  V3f operator+(const V3f &rhs) const {
    V3f r(*this);
    r += rhs;
    return r;
  }
  V3f operator*(const float rhs) const {
    V3f r(*this);
    r *= rhs;
    return r;
  }
  auto square() const {
    auto s = float{};
    for (auto i : range(p.size())) s += p[i] * p[i];
    return s;
  }
  auto len() const { return sqrtf(square()); }
  V3f() = default;
  V3f(const V3f &) = default;
  V3f &operator=(const V3f &) = default;
  V3f &operator=(V3f &&) = default;
  V3f(V3f &&) = default;
  V3f(float x, float y, float z) : p{x, y, z} {};
  friend V3f cross(const V3f &lhs, const V3f &rhs) {
    auto &a = lhs.p;
    auto &b = rhs.p;
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }
};

const size_t sz = 200;
const float L0 = 1.0f / sz;
const float L1 = L0 + L0;
const float D0 = sqrt(L0 * L0 + L0 * L0);
const float K = 1000.0;
const float KD = 1000.0;
const float MasseSurf = 0.3f;
float dt = 1e-4 / (sz / 70.0);

struct Node {
  using Direction = V3f;
  using Position = V3f;
  using Speed = V3f;
  using Force = V3f;
  Position position;
  Speed speed;
  Force f;
  Direction normal;
  float masse = MasseSurf / (sz * sz);
  float moveability = (sz * sz) / MasseSurf;  // mass inverse
};

double compute_time = {};

static std::array<std::array<Node, sz>, sz> napkin;

// GPU resources
GLuint vboPos = 0, vboNorm = 0, vao = 0, shaderProgram = 0;
GLint uModelView = -1, uProjection = -1, uLightPos = -1, uLightColor = -1;
size_t vertexCount = 6 * (sz - 1) * (sz - 1);
static std::vector<V3f> gpuPos;
static std::vector<V3f> gpuNorm;

static GLuint compileShader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    std::cerr << log << std::endl;
  }
  return s;
}

static GLuint createProgram(const char* vs, const char* fs) {
  GLuint v = compileShader(GL_VERTEX_SHADER, vs);
  GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glBindAttribLocation(p, 0, "position");
  glBindAttribLocation(p, 1, "normal");
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetProgramInfoLog(p, sizeof(log), nullptr, log);
    std::cerr << log << std::endl;
  }
  glDeleteShader(v);
  glDeleteShader(f);
  return p;
}

static const char* vsSrc = R"(
#version 120
attribute vec3 position;
attribute vec3 normal;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
varying vec3 vNormal;
varying vec3 vPos;
void main() {
  vec4 pos = modelViewMatrix * vec4(position, 1.0);
  vPos = pos.xyz;
  vNormal = mat3(modelViewMatrix) * normal;
  gl_Position = projectionMatrix * pos;
}
)";

static const char* fsSrc = R"(
#version 120
varying vec3 vNormal;
varying vec3 vPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
void main() {
  vec3 N = normalize(vNormal);
  vec3 L = normalize(lightPos - vPos);
  float diff = max(dot(N, L), 0.0);
  gl_FragColor = vec4(lightColor * diff, 1.0);
}
)";

float angle = 30;

/* GLUT callback Handlers */
static int w = 400, h = 400;

static V3f camera(-3.0f, -3.0f, 3.0f);

const GLfloat light_ambient[] = {0.0f, 0.0f, 0.0f, 1.0f};
const GLfloat light_diffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat light_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat light_position[] = {2.0f, 5.0f, 5.0f, 0.0f};

const GLfloat mat_ambient[] = {0.7f, 0.7f, 0.7f, 1.0f};
const GLfloat mat_diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
const GLfloat mat_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat high_shininess[] = {100.0f};

static void do_resize() {
  const float ar = (float)w / (float)h;

  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(angle, ar, 0.1, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(camera.p[0], camera.p[1], camera.p[2], 0.0f, 0.0f, 0.0f, 0, 0, 1);

  std::cout << "camera " << camera.p[0] << " " << camera.p[1] << " "
            << camera.p[2] << std::endl;
  // glutPostRedisplay();
}

static void resize(int width, int height) {
  w = width;
  h = height;
  do_resize();
}

void build_norm() {
  auto addnorm = [](Node &n1, Node &n2, Node &n3) {
    auto n = cross(n2.position - n1.position, n3.position - n1.position);
    n1.normal += n;
    n2.normal += n;
    n3.normal += n;
  };
  for (auto i : range(sz))
    for (auto j : range(sz)) napkin[i][j].normal = V3f{};
  for (auto i : range(sz - 1)) {
    for (auto j : range(sz - 1)) {
      addnorm(napkin[i + 0][j + 0], napkin[i + 0][j + 1], napkin[i + 1][j + 0]);
      addnorm(napkin[i + 0][j + 1], napkin[i + 1][j + 1], napkin[i + 0][j + 0]);
      addnorm(napkin[i + 1][j + 1], napkin[i + 1][j + 0], napkin[i + 0][j + 1]);
      addnorm(napkin[i + 1][j + 0], napkin[i + 0][j + 0], napkin[i + 1][j + 1]);
    }
  }

  for (auto i : range(sz))
    for (auto j : range(sz)) {
      auto n = napkin[i][j].normal;
      n *= -1.0f / n.len();
      napkin[i][j].normal = n;
    }
}

static void updateBuffers() {
  size_t idx = 0;
  for (auto i : range(sz - 1)) {
    for (auto j : range(sz - 1)) {
      const Node &n00 = napkin[i][j];
      const Node &n10 = napkin[i + 1][j];
      const Node &n01 = napkin[i][j + 1];
      const Node &n11 = napkin[i + 1][j + 1];

      gpuPos[idx] = n00.position;
      gpuNorm[idx++] = n00.normal;
      gpuPos[idx] = n10.position;
      gpuNorm[idx++] = n10.normal;
      gpuPos[idx] = n01.position;
      gpuNorm[idx++] = n01.normal;

      gpuPos[idx] = n11.position;
      gpuNorm[idx++] = n11.normal;
      gpuPos[idx] = n01.position;
      gpuNorm[idx++] = n01.normal;
      gpuPos[idx] = n10.position;
      gpuNorm[idx++] = n10.normal;
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vboPos);
  glBufferSubData(GL_ARRAY_BUFFER, 0, gpuPos.size() * sizeof(V3f),
                  gpuPos.data());
  glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
  glBufferSubData(GL_ARRAY_BUFFER, 0, gpuNorm.size() * sizeof(V3f),
                  gpuNorm.data());
}

double draw_time = {};

static void display(void) {
  const auto t = glutGet(GLUT_ELAPSED_TIME);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  build_norm();
  updateBuffers();

  glUseProgram(shaderProgram);
  glPushMatrix();
  auto &p = napkin[sz / 2][sz / 2].position.p;
  glTranslatef(-p[0], -p[1], -p[2]);

  GLfloat mv[16];
  GLfloat pr[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, mv);
  glGetFloatv(GL_PROJECTION_MATRIX, pr);

  glUniformMatrix4fv(uModelView, 1, GL_FALSE, mv);
  glUniformMatrix4fv(uProjection, 1, GL_FALSE, pr);
  glUniform3fv(uLightPos, 1, light_position);
  glUniform3fv(uLightColor, 1, light_diffuse);

  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, vertexCount);
  glBindVertexArray(0);

  glPopMatrix();
  glUseProgram(0);

  //       std::cout<<"draw"<< napkin[sz-1][sz-1].position.p[0] << " " <<
  //       napkin[sz-1][sz-1].position.p[1] << " " <<
  //       napkin[sz-1][sz-1].position.p[2] <<std::endl;

  glutSwapBuffers();
  const auto t1 = glutGet(GLUT_ELAPSED_TIME);
  draw_time += t1 - t;
}

static int lastx, lasty;

void motion(int x, int y) {
  int dx = x - lastx;
  int dy = y - lasty;
  // camera;
  V3f h(0.0, 0.0, 1.0);
  V3f c(0.0, 0.0, 0.0);
  V3f v = c - camera;
  auto horiz = cross(v, h);
  auto top = cross(horiz, v);
  horiz *= dx * 0.01 / horiz.len();
  top *= dy * 0.01 / top.len();
  auto l = v.len();
  v += horiz;
  v += top;
  v *= l / v.len();
  camera = c - v;
  do_resize();
  std::cout << "camera " << camera.p[0] << " " << camera.p[1] << " "
            << camera.p[2] << std::endl;

  lastx = x;
  lasty = y;
}

void mouse(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    glutMotionFunc(motion);
  } else {
    glutMotionFunc(nullptr);
  }
  lastx = x;
  lasty = y;
}

static void key(unsigned char key, int x, int y) {
  switch (key) {
    case 27:
    case 'q':
      exit(0);
      break;

    case '+':
      angle /= 1.05;
      do_resize();
      break;

    case '-':
      angle *= 1.05;
      do_resize();
      break;

    case '>':
      dt *= 1.1;
      std::cout << "dt=" << dt << std::endl;
      break;
    case '<':
      dt /= 1.1;
      std::cout << "dt=" << dt << std::endl;
      break;
  }

  glutPostRedisplay();
}

size_t tasks = 8;

static void idle(void)

{
  using Forces = std::array<V3f, sz * sz>;
  const auto t = glutGet(GLUT_ELAPSED_TIME);
  std::atomic<int> num = {};

  std::vector<Forces *> allforcestb{tasks};

  auto doit = [&allforcestb, &num](size_t m, size_t r) {
    assert(r < m);
    static thread_local Forces *pforces = {};
    if (!pforces) pforces = new Forces{};
    Forces &forces = *pforces;
    allforcestb[num++] = pforces;
    const auto addForce = [&forces](Node &n1, Node &n2, const float L0,
                                    const float K) {
      auto v = n2.position - n1.position;
      auto l = v.len() / L0;
      v *= K * (l - 1.0);
      auto &n0 = napkin[0][0];
      forces[&n1 - &n0] += v;
      forces[&n2 - &n0] -= v;
    };

    for (auto i : range(sz))
      if (i % m == r) {
        for (auto j : range(sz - 1)) {
          addForce(napkin[i][j], napkin[i][j + 1], L0, K);
          addForce(napkin[j][i], napkin[j + 1][i], L0, K);
        }
      }
    for (auto i : range(sz - 1))
      if (i % m == r) {
        for (auto j : range(sz - 1)) {
          addForce(napkin[i][j], napkin[i + 1][j + 1], D0, KD);
          addForce(napkin[i + 1][j], napkin[i][j + 1], D0, KD);
        }
      }
    for (auto i : range(sz))
      if (i % m == r) {
        for (auto j : range(sz - 2)) {
          addForce(napkin[i][j], napkin[i][j + 2], L1, KD);
          addForce(napkin[j][i], napkin[j + 2][i], L1, KD);
        }
      }
  };

  using AllForces = std::unordered_set<Forces *>;
  AllForces allforces;
#if 0
    for(auto task : range(tasks))
        doit(tasks,task);

#else
  std::vector<MyNamespace::ThreadPool::TaskFuture<void> > promises;

  for (auto task : range(tasks)) {
    auto f = MyNamespace::DefaultThreadPool::submitJob(doit, tasks, task);
    //        auto f=std::async( std::launch::async ,doit,tasks,task);
    promises.emplace_back(std::move(f));
  }

  for (auto &p : promises) {
    p.get();
  };
#endif
  for (auto f : allforcestb)
    if (f) allforces.emplace(f);

  std::cout << "nb_threads " << allforces.size() << std::endl;

  auto step = [&allforces, &num](size_t m, size_t r) {
    assert(r < m);
    const auto damp = exp(-1.2 * dt);

    for (auto i : range(sz))
      if (i % m == r) {
        for (auto j : range(sz)) {
          auto &n = napkin[i][j];
          n.speed *= damp;

          auto &n0 = napkin[0][0];
          const auto i_f = &n - &n0;
          V3f f(0.0f, 0.0f, 9.81f * n.masse);
          for (auto &fs : allforces) {
            auto &ff = (*fs)[i_f];
            f += ff;
            ff = V3f{};
          }
          n.speed += f * n.moveability * dt;
          n.position += n.speed * dt;
        }
      }
  };
  {
    std::vector<MyNamespace::ThreadPool::TaskFuture<void> > promises;

    for (auto task : range(tasks)) {
      auto f = MyNamespace::DefaultThreadPool::submitJob(step, tasks, task);
      promises.emplace_back(std::move(f));
    }

    for (auto &p : promises) {
      p.get();
    };
  }

  //  std::cout<<"tick"<< napkin[sz-1][sz-1].position.p[0] << " " <<
  //  napkin[sz-1][sz-1].position.p[1] << " " <<
  //  napkin[sz-1][sz-1].position.p[2] <<std::endl;
  const auto t1 = glutGet(GLUT_ELAPSED_TIME);
  compute_time += t1 - t;

  if (compute_time >
      draw_time * 50.0)  // throw a redraw if draw time is 2% of comput. time
    glutPostRedisplay();
}


/* Program entry point */

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);

  for(int i=1;i<argc;i++)
    tasks=atoi(argv[i]);
  glutInitWindowSize(640, 480);
  glutInitWindowPosition(10, 10);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

  glutCreateWindow("GLUT Shapes");

  glutReshapeFunc(resize);
  glutDisplayFunc(display);
  glutKeyboardFunc(key);
  glutMouseFunc(mouse);

  glutIdleFunc(idle);

  glClearColor(1, 1, 1, 1);
  // glEnable(GL_CULL_FACE);
  // glCullFace(GL_BACK);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  glEnable(GL_NORMALIZE);

  gpuPos.resize(vertexCount);
  gpuNorm.resize(vertexCount);

  shaderProgram = createProgram(vsSrc, fsSrc);
  uModelView = glGetUniformLocation(shaderProgram, "modelViewMatrix");
  uProjection = glGetUniformLocation(shaderProgram, "projectionMatrix");
  uLightPos = glGetUniformLocation(shaderProgram, "lightPos");
  uLightColor = glGetUniformLocation(shaderProgram, "lightColor");

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vboPos);
  glBindBuffer(GL_ARRAY_BUFFER, vboPos);
  glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(V3f), nullptr,
               GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(V3f), (void *)0);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &vboNorm);
  glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
  glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(V3f), nullptr,
               GL_DYNAMIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(V3f), (void *)0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);


  V3f c(0.5, 0.5, 0.0);
  for (auto i : range(sz))
    for (auto j : range(sz)) {
      auto &n = napkin[i][j];
      n.position.p[0] = i * L0;
      n.position.p[1] = j * L0;
      if ((n.position - c).len() < 0.2) n.moveability = 0.0;
    }
  //    napkin[0][0].moveability=0.0;
  //    napkin[sz-1][0].moveability=0.0;

  glutMainLoop();

  return EXIT_SUCCESS;
}

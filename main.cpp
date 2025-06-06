
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

float angle = 30;

/* GLUT callback Handlers */
static int w = 400, h = 400;

static V3f camera(-3.0f, -3.0f, 3.0f);

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

double draw_time = {};

static void display(void) {
  const auto t = glutGet(GLUT_ELAPSED_TIME);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glColor3d(1, 0, 0);

  {
    // glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    build_norm();

    glPushMatrix();
    auto &p = napkin[sz / 2][sz / 2].position.p;

    //        glTranslated(0,1.2,-6);
    glTranslatef(-p[0], -p[1], -p[2]);
    //        glScalef(0.5f,0.5f,0.5f);
    glBegin(GL_TRIANGLES);

    auto outNode = [](const Node &n) {
      glNormal3fv(n.normal.p.data());
      glVertex3fv(n.position.p.data());
    };
    for (auto i : range(sz - 1)) {
      for (auto j : range(sz - 1)) {
        outNode(napkin[i + 0][j + 0]);
        outNode(napkin[i + 1][j + 0]);
        outNode(napkin[i + 0][j + 1]);

        outNode(napkin[i + 1][j + 1]);
        outNode(napkin[i + 0][j + 1]);
        outNode(napkin[i + 1][j + 0]);
      }
    }
    glEnd();
    glPopMatrix();

    //       std::cout<<"draw"<< napkin[sz-1][sz-1].position.p[0] << " " <<
    //       napkin[sz-1][sz-1].position.p[1] << " " <<
    //       napkin[sz-1][sz-1].position.p[2] <<std::endl;
  }

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

const GLfloat light_ambient[] = {0.0f, 0.0f, 0.0f, 1.0f};
const GLfloat light_diffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat light_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat light_position[] = {2.0f, 5.0f, 5.0f, 0.0f};

const GLfloat mat_ambient[] = {0.7f, 0.7f, 0.7f, 1.0f};
const GLfloat mat_diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
const GLfloat mat_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
const GLfloat high_shininess[] = {100.0f};

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

  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_LIGHTING);

  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);

  glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
  glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

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

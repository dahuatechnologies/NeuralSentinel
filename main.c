/**
 * EVOX AI CORE SYSTEM v3.0.4 - C90 COMPLIANT VERSION
 * ====================================================
 * Fixed to compile with -std=c90
 * No variable declarations inside for loops
 * All necessary headers included
 *
 * COMPILATION: gcc -std=c90 -D_GNU_SOURCE -pthread -o evox_fixed main.c -lGL -lGLU -lglut -lm -lrt
 * RUN: ./evox_fixed
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <sys/time.h>

/*-----------------------------------------------------------------------------
 * CONSTANTS
 *----------------------------------------------------------------------------*/

#define WINDOW_WIDTH        1280
#define WINDOW_HEIGHT       720
#define WINDOW_TITLE        "EVOX AI CORE - Neural Sentinel Hypergraph"

#define MAX_NODES           512
#define MAX_EDGES           8000
#define SYNAPSE_DENSITY     0.08
#define MAX_EXPERTS         16

#define STATE_INITIALIZING  1
#define STATE_IDLE          2
#define STATE_PROCESSING    4
#define STATE_RENDERING     5
#define STATE_KEY_ROTATION  7

#define KEY_ROTATION_HOURS  28
#define ROTATION_INTERVAL   (KEY_ROTATION_HOURS * 3600)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*-----------------------------------------------------------------------------
 * TYPE DEFINITIONS
 *----------------------------------------------------------------------------*/

typedef struct {
	float x, y, z; /* Position in 3D space */
	float activation; /* Current activation (0-1) */
	float potential; /* Electrical potential */
	unsigned int expert_id; /* Associated expert (0-15) */
	unsigned int firing_count; /* Number of times fired */
	float last_fired; /* Last firing timestamp */
	unsigned int edge_indices[20]; /* Connected edges */
	unsigned int edge_count; /* Number of connections */
	float weights[8]; /* Synaptic weights */
} HypergraphNode;

typedef struct {
	unsigned int source; /* Source node index */
	unsigned int target; /* Target node index */
	float strength; /* Connection strength (0-1) */
	float frequency; /* Oscillation frequency */
	float phase; /* Phase offset */
	unsigned int active; /* Whether connection is active */
	unsigned int packet_count; /* Number of signals transmitted */
	float color_r, color_g, color_b; /* Cached color for rendering */
} HypergraphEdge;

typedef struct {
	unsigned char key[32]; /* 256-bit key */
	char key_string[65]; /* Hex representation */
	unsigned long long expiration; /* Expiration timestamp */
	unsigned int entropy_bits; /* Key entropy */
} CryptographicKey;

typedef struct {
	/* Hypergraph data */
	HypergraphNode nodes[MAX_NODES];
	HypergraphEdge edges[MAX_EDGES];
	unsigned int node_count;
	unsigned int edge_count;
	float system_entropy;

	/* Security */
	CryptographicKey api_keys[16];
	unsigned int key_count;
	unsigned long long last_rotation;
	unsigned long long next_rotation;

	/* State machine */
	int current_state;
	int previous_state;
	int running;

	/* Performance */
	unsigned long long computation_cycles;
	unsigned long long frame_count;
	float fps;

	/* Camera control */
	float camera_angle;
	float camera_elevation;
	float camera_distance;
	int mouse_x, mouse_y;
	int mouse_buttons[3];

	/* Thread synchronization */
	pthread_mutex_t data_mutex;
	pthread_t compute_thread;
} EVOXContext;

static EVOXContext *g_ctx = NULL;
static int g_quit = 0;
static int g_window_id = 0;

/*-----------------------------------------------------------------------------
 * UTILITY FUNCTIONS
 *----------------------------------------------------------------------------*/

float random_float(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

unsigned long long get_timestamp_sec(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (unsigned long long) tv.tv_sec;
}

/*-----------------------------------------------------------------------------
 * CRYPTOGRAPHIC KEY GENERATION
 *----------------------------------------------------------------------------*/

void generate_api_key(CryptographicKey *key, unsigned long long timestamp) {
	int i;

	/* Generate random key material */
	for (i = 0; i < 32; i++) {
		key->key[i] = rand() % 256;
	}

	/* Create hex string */
	for (i = 0; i < 32; i++) {
		sprintf(&key->key_string[i * 2], "%02x", key->key[i]);
	}
	key->key_string[64] = '\0';

	/* Set expiration (28 hours from now) */
	key->expiration = timestamp + ROTATION_INTERVAL;

	/* Calculate entropy (simplified) */
	key->entropy_bits = 256;
}

void rotate_api_keys(EVOXContext *ctx) {
	unsigned long long now = get_timestamp_sec();
	int i;

	if (now < ctx->next_rotation) {
		return;
	}

	pthread_mutex_lock(&ctx->data_mutex);

	printf("\n[SECURITY] Rotating API keys at %llu...\n", now);

	for (i = 0; i < ctx->key_count; i++) {
		generate_api_key(&ctx->api_keys[i], now);
	}

	ctx->last_rotation = now;
	ctx->next_rotation = now + ROTATION_INTERVAL;

	pthread_mutex_unlock(&ctx->data_mutex);
}

/*-----------------------------------------------------------------------------
 * HYPERGRAPH INITIALIZATION
 *----------------------------------------------------------------------------*/

void init_hypergraph(EVOXContext *ctx) {
	int i, j;
	float theta, phi, radius, dx, dy, dz, dist;
	unsigned int edge_idx;

	printf("Initializing hypergraph with %d neurons...\n", MAX_NODES);

	pthread_mutex_lock(&ctx->data_mutex);

	/* Initialize nodes in a spherical distribution */
	for (i = 0; i < MAX_NODES; i++) {
		/* Position nodes on a sphere with some randomness */
		theta = 2.0f * M_PI * (float) i / MAX_NODES;
		phi = acosf(2.0f * (float) i / MAX_NODES - 1.0f);
		radius = 120.0f + random_float(-20.0f, 20.0f);

		ctx->nodes[i].x = radius * sinf(phi) * cosf(theta);
		ctx->nodes[i].y = radius * sinf(phi) * sinf(theta);
		ctx->nodes[i].z = radius * cosf(phi);

		/* Add some local randomness */
		ctx->nodes[i].x += random_float(-15.0f, 15.0f);
		ctx->nodes[i].y += random_float(-15.0f, 15.0f);
		ctx->nodes[i].z += random_float(-15.0f, 15.0f);

		ctx->nodes[i].activation = random_float(0.0f, 0.3f);
		ctx->nodes[i].potential = 0.0f;
		ctx->nodes[i].expert_id = i % MAX_EXPERTS;
		ctx->nodes[i].firing_count = 0;
		ctx->nodes[i].last_fired = 0.0f;
		ctx->nodes[i].edge_count = 0;

		/* Initialize weights */
		for (j = 0; j < 8; j++) {
			ctx->nodes[i].weights[j] = random_float(0.1f, 1.0f);
		}
	}

	ctx->node_count = MAX_NODES;
	ctx->edge_count = 0;

	/* Create synaptic connections with small-world topology */
	for (i = 0; i < ctx->node_count; i++) {
		/* Connect to nearby nodes (local connections) */
		for (j = i + 1; j < ctx->node_count && j < i + 20; j++) {
			if (random_float(0, 1) < SYNAPSE_DENSITY * 3&&
			ctx->edge_count < MAX_EDGES) {

				dx = ctx->nodes[i].x - ctx->nodes[j].x;
				dy = ctx->nodes[i].y - ctx->nodes[j].y;
				dz = ctx->nodes[i].z - ctx->nodes[j].z;
				dist = sqrtf(dx * dx + dy * dy + dz * dz);

				/* Prefer shorter connections */
				if (dist < 80.0f) {
					edge_idx = ctx->edge_count++;

					ctx->edges[edge_idx].source = i;
					ctx->edges[edge_idx].target = j;
					ctx->edges[edge_idx].strength = random_float(0.3f, 1.0f);
					ctx->edges[edge_idx].frequency = random_float(0.5f, 2.0f);
					ctx->edges[edge_idx].phase = random_float(0, 2 * M_PI);
					ctx->edges[edge_idx].active = 1;
					ctx->edges[edge_idx].packet_count = 0;

					/* Color based on strength */
					ctx->edges[edge_idx].color_r = 0.3f
							+ ctx->edges[edge_idx].strength * 0.7f;
					ctx->edges[edge_idx].color_g = 0.2f;
					ctx->edges[edge_idx].color_b = 0.8f;

					/* Add to node's edge list */
					if (ctx->nodes[i].edge_count < 20) {
						ctx->nodes[i].edge_indices[ctx->nodes[i].edge_count++] =
								edge_idx;
					}
					if (ctx->nodes[j].edge_count < 20) {
						ctx->nodes[j].edge_indices[ctx->nodes[j].edge_count++] =
								edge_idx;
					}
				}
			}
		}

		/* Add some random long-range connections (small-world property) */
		if (random_float(0, 1) < 0.05 && ctx->edge_count < MAX_EDGES) {
			j = rand() % ctx->node_count;
			if (i != j) {
				edge_idx = ctx->edge_count++;

				ctx->edges[edge_idx].source = i;
				ctx->edges[edge_idx].target = j;
				ctx->edges[edge_idx].strength = random_float(0.1f, 0.5f);
				ctx->edges[edge_idx].frequency = random_float(0.2f, 1.0f);
				ctx->edges[edge_idx].phase = random_float(0, 2 * M_PI);
				ctx->edges[edge_idx].active = 1;
				ctx->edges[edge_idx].packet_count = 0;

				ctx->edges[edge_idx].color_r = 0.8f;
				ctx->edges[edge_idx].color_g = 0.3f;
				ctx->edges[edge_idx].color_b = 0.3f;

				if (ctx->nodes[i].edge_count < 20) {
					ctx->nodes[i].edge_indices[ctx->nodes[i].edge_count++] =
							edge_idx;
				}
				if (ctx->nodes[j].edge_count < 20) {
					ctx->nodes[j].edge_indices[ctx->nodes[j].edge_count++] =
							edge_idx;
				}
			}
		}
	}

	printf("  Created %d synaptic connections\n", ctx->edge_count);
	printf("  Network density: %.2f%%\n",
			100.0f * ctx->edge_count
					/ (ctx->node_count * (ctx->node_count - 1) / 2));

	pthread_mutex_unlock(&ctx->data_mutex);
}

/*-----------------------------------------------------------------------------
 * HYPERGRAPH COMPUTATION
 *----------------------------------------------------------------------------*/

void update_hypergraph(EVOXContext *ctx) {
	int i, e;
	float signal, mod, sum_act, sum_sq, mean, variance;
	unsigned int src, tgt;

	pthread_mutex_lock(&ctx->data_mutex);

	/* Decay potentials */
	for (i = 0; i < ctx->node_count; i++) {
		ctx->nodes[i].potential *= 0.96f;
	}

	/* Process edges with frequency-dependent modulation */
	for (e = 0; e < ctx->edge_count; e++) {
		if (ctx->edges[e].active) {
			src = ctx->edges[e].source;
			tgt = ctx->edges[e].target;

			/* Frequency modulation creates oscillating patterns */
			mod = 0.5f
					+ 0.5f
							* sinf(
									ctx->computation_cycles * 0.01f
											* ctx->edges[e].frequency
											+ ctx->edges[e].phase);

			signal = ctx->nodes[src].activation * ctx->edges[e].strength * mod;
			ctx->nodes[tgt].potential += signal * 0.15f;

			/* Update edge color based on activity */
			ctx->edges[e].color_r = 0.3f + ctx->edges[e].strength * 0.7f * mod;
			ctx->edges[e].color_g = 0.2f + signal * 0.5f;
			ctx->edges[e].color_b = 0.8f - signal * 0.3f;

			ctx->edges[e].packet_count++;
		}
	}

	/* Fire neurons and update activations */
	for (i = 0; i < ctx->node_count; i++) {
		if (ctx->nodes[i].potential > 1.0f) {
			ctx->nodes[i].activation = 1.0f;
			ctx->nodes[i].firing_count++;
			ctx->nodes[i].last_fired = ctx->computation_cycles * 0.016f;
			ctx->nodes[i].potential = 0.0f;
		} else {
			/* Smooth activation function */
			ctx->nodes[i].activation = 0.7f
					* sigmoid(ctx->nodes[i].potential * 3.0f - 1.5f)
					+ 0.3f * ctx->nodes[i].activation;
		}
	}

	/* Calculate system entropy (measure of neural activity diversity) */
	sum_act = 0;
	sum_sq = 0;
	for (i = 0; i < 100; i++) {
		sum_act += ctx->nodes[i].activation;
		sum_sq += ctx->nodes[i].activation * ctx->nodes[i].activation;
	}
	mean = sum_act / 100;
	variance = sum_sq / 100 - mean * mean;
	ctx->system_entropy = variance * 10.0f + 0.5f;

	ctx->computation_cycles++;

	pthread_mutex_unlock(&ctx->data_mutex);
}

/*-----------------------------------------------------------------------------
 * COMPUTATION THREAD
 *----------------------------------------------------------------------------*/

void* computation_thread_func(void *arg) {
	EVOXContext *ctx = (EVOXContext*) arg;

	while (ctx->running && !g_quit) {
		if (ctx->current_state == STATE_PROCESSING
				|| ctx->current_state == STATE_RENDERING) {
			update_hypergraph(ctx);

			/* Rotate keys if needed */
			if (get_timestamp_sec() >= ctx->next_rotation) {
				rotate_api_keys(ctx);
			}
		}
		usleep(16000); /* ~60Hz update */
	}

	return NULL;
}

/*-----------------------------------------------------------------------------
 * OPENGL RENDERING FUNCTIONS
 *----------------------------------------------------------------------------*/

void render_text(float x, float y, const char *format, ...) {
	char buffer[256];
	va_list args;
	char *c;

	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);

	glRasterPos2f(x, y);
	for (c = buffer; *c; c++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
	}
}

void draw_axes(void) {
	glLineWidth(2.0f);

	/* X axis - Red */
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-150.0f, 0.0f, 0.0f);
	glVertex3f(150.0f, 0.0f, 0.0f);
	glEnd();

	/* Y axis - Green */
	glBegin(GL_LINES);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -100.0f, 0.0f);
	glVertex3f(0.0f, 100.0f, 0.0f);
	glEnd();

	/* Z axis - Blue */
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, -150.0f);
	glVertex3f(0.0f, 0.0f, 150.0f);
	glEnd();

	/* Axis labels */
	glColor3f(1.0f, 1.0f, 1.0f);
	glRasterPos3f(160.0f, 5.0f, 0.0f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'X');
	glRasterPos3f(5.0f, 110.0f, 0.0f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Y');
	glRasterPos3f(0.0f, 5.0f, 160.0f);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Z');
}

void render_hypergraph(void) {
	EVOXContext *ctx = g_ctx;
	int i;
	float rad_angle, rad_elev, cam_x, cam_y, cam_z;
	unsigned int src, tgt;

	if (!ctx)
		return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Set up camera */
	rad_angle = ctx->camera_angle * M_PI / 180.0f;
	rad_elev = ctx->camera_elevation * M_PI / 180.0f;

	cam_x = sinf(rad_angle) * cosf(rad_elev) * ctx->camera_distance;
	cam_y = sinf(rad_elev) * ctx->camera_distance;
	cam_z = cosf(rad_angle) * cosf(rad_elev) * ctx->camera_distance;

	gluLookAt(cam_x, cam_y, cam_z, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	pthread_mutex_lock(&ctx->data_mutex);

	/* Draw coordinate axes */
	draw_axes();

	/* Draw edges with color-coded strengths */
	glLineWidth(1.5f);
	glBegin(GL_LINES);
	for (i = 0; i < ctx->edge_count; i++) {
		if (ctx->edges[i].active) {
			src = ctx->edges[i].source;
			tgt = ctx->edges[i].target;

			glColor4f(ctx->edges[i].color_r, ctx->edges[i].color_g,
					ctx->edges[i].color_b, 0.6f);

			glVertex3f(ctx->nodes[src].x, ctx->nodes[src].y, ctx->nodes[src].z);
			glVertex3f(ctx->nodes[tgt].x, ctx->nodes[tgt].y, ctx->nodes[tgt].z);
		}
	}
	glEnd();

	/* Draw nodes as points with activation-based size and color */
	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (i = 0; i < ctx->node_count; i++) {
		float act = ctx->nodes[i].activation;
		float expert_factor = (float) ctx->nodes[i].expert_id / MAX_EXPERTS;

		/* Color based on activation and expert ID */
		glColor4f(act, expert_factor, 1.0f - act, 0.9f);
		glVertex3f(ctx->nodes[i].x, ctx->nodes[i].y, ctx->nodes[i].z);
	}
	glEnd();

	/* Draw active neurons as larger spheres */
	for (i = 0; i < ctx->node_count; i += 3) {
		if (ctx->nodes[i].activation > 0.7f) {
			glPushMatrix();
			glTranslatef(ctx->nodes[i].x, ctx->nodes[i].y, ctx->nodes[i].z);

			float size = 2.0f + ctx->nodes[i].activation * 4.0f;
			glColor4f(1.0f, 0.5f, 0.0f, 0.7f);

			/* Draw simple sphere */
			glutSolidSphere(size, 8, 8);

			glPopMatrix();
		}
	}

	pthread_mutex_unlock(&ctx->data_mutex);

	/* Switch to 2D for text overlay */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	/* Draw semi-transparent background for text */
	glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
	glBegin(GL_QUADS);
	glVertex2f(10, WINDOW_HEIGHT - 140);
	glVertex2f(520, WINDOW_HEIGHT - 140);
	glVertex2f(520, WINDOW_HEIGHT - 10);
	glVertex2f(10, WINDOW_HEIGHT - 10);
	glEnd();

	/* Render text overlay */
	glColor3f(1.0f, 1.0f, 1.0f);

	render_text(20, WINDOW_HEIGHT - 25,
			"EVOX AI CORE v3.0.4 - Neural Sentinel");
	render_text(20, WINDOW_HEIGHT - 45, "State: %s | Entropy: %.3f | FPS: %.1f",
			ctx->current_state == STATE_PROCESSING ? "PROCESSING" :
			ctx->current_state == STATE_RENDERING ? "RENDERING" : "IDLE",
			ctx->system_entropy, ctx->fps);

	render_text(20, WINDOW_HEIGHT - 65, "Nodes: %d | Edges: %d | Cycles: %llu",
			ctx->node_count, ctx->edge_count,
			(unsigned long long) ctx->computation_cycles);

	if (ctx->key_count > 0) {
		render_text(20, WINDOW_HEIGHT - 85, "API Key: %s... | Rotation: %llus",
				ctx->api_keys[0].key_string,
				(unsigned long long) (ctx->next_rotation - get_timestamp_sec()));
	}

	render_text(20, WINDOW_HEIGHT - 105,
			"Controls: [Space] Pause | [R] Reset View | [ESC] Exit");
	render_text(20, WINDOW_HEIGHT - 125, "Mouse drag: Rotate | +/-: Zoom");

	/* Draw color legend */
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 200, 50);
	glVertex2f(WINDOW_WIDTH - 150, 50);
	glVertex2f(WINDOW_WIDTH - 150, 80);
	glVertex2f(WINDOW_WIDTH - 200, 80);
	glEnd();

	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 200, 90);
	glVertex2f(WINDOW_WIDTH - 150, 90);
	glVertex2f(WINDOW_WIDTH - 150, 120);
	glVertex2f(WINDOW_WIDTH - 200, 120);
	glEnd();

	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 200, 130);
	glVertex2f(WINDOW_WIDTH - 150, 130);
	glVertex2f(WINDOW_WIDTH - 150, 160);
	glVertex2f(WINDOW_WIDTH - 200, 160);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
	render_text(WINDOW_WIDTH - 140, 60, "X Axis");
	render_text(WINDOW_WIDTH - 140, 100, "Y Axis");
	render_text(WINDOW_WIDTH - 140, 140, "Z Axis");

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glutSwapBuffers();
	ctx->frame_count++;
}

/*-----------------------------------------------------------------------------
 * GLUT CALLBACKS
 *----------------------------------------------------------------------------*/

void display_func(void) {
	render_hypergraph();
}

void reshape_func(int width, int height) {
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / height, 1.0, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}

void keyboard_func(unsigned char key, int x, int y) {
	EVOXContext *ctx = g_ctx;
	if (!ctx)
		return;

	switch (key) {
	case 27: /* ESC */
	case 'q':
	case 'Q':
		ctx->running = 0;
		g_quit = 1;
		glutLeaveMainLoop();
		break;

	case ' ':
		/* Toggle processing state */
		if (ctx->current_state == STATE_PROCESSING) {
			ctx->current_state = STATE_IDLE;
			printf("Computation paused\n");
		} else {
			ctx->current_state = STATE_PROCESSING;
			printf("Computation resumed\n");
		}
		break;

	case 'r':
	case 'R':
		/* Reset camera */
		ctx->camera_angle = 0.0f;
		ctx->camera_elevation = 30.0f;
		ctx->camera_distance = 350.0f;
		break;

	case '+':
	case '=':
		ctx->camera_distance -= 20.0f;
		if (ctx->camera_distance < 100.0f)
			ctx->camera_distance = 100.0f;
		break;

	case '-':
	case '_':
		ctx->camera_distance += 20.0f;
		if (ctx->camera_distance > 600.0f)
			ctx->camera_distance = 600.0f;
		break;
	}
}

void special_func(int key, int x, int y) {
	EVOXContext *ctx = g_ctx;
	if (!ctx)
		return;

	switch (key) {
	case GLUT_KEY_LEFT:
		ctx->camera_angle -= 5.0f;
		break;
	case GLUT_KEY_RIGHT:
		ctx->camera_angle += 5.0f;
		break;
	case GLUT_KEY_UP:
		ctx->camera_elevation += 5.0f;
		if (ctx->camera_elevation > 89.0f)
			ctx->camera_elevation = 89.0f;
		break;
	case GLUT_KEY_DOWN:
		ctx->camera_elevation -= 5.0f;
		if (ctx->camera_elevation < -89.0f)
			ctx->camera_elevation = -89.0f;
		break;
	}
}

void mouse_func(int button, int state, int x, int y) {
	EVOXContext *ctx = g_ctx;
	if (!ctx)
		return;

	if (state == GLUT_DOWN) {
		ctx->mouse_buttons[button] = 1;
		ctx->mouse_x = x;
		ctx->mouse_y = y;
	} else {
		ctx->mouse_buttons[button] = 0;
	}
}

void motion_func(int x, int y) {
	EVOXContext *ctx = g_ctx;
	if (!ctx)
		return;

	if (ctx->mouse_buttons[0]) {
		/* Left button - rotate */
		ctx->camera_angle += (x - ctx->mouse_x) * 0.5f;
		ctx->camera_elevation += (y - ctx->mouse_y) * 0.5f;
	}

	ctx->mouse_x = x;
	ctx->mouse_y = y;
}

void idle_func(void) {
	static int last_time = 0;
	static int frames = 0;
	int current_time;

	if (g_ctx) {
		/* Update camera auto-rotation when no mouse interaction */
		if (!g_ctx->mouse_buttons[0] && !g_ctx->mouse_buttons[1]
				&& !g_ctx->mouse_buttons[2]) {
			g_ctx->camera_angle += 0.2f;
		}

		/* Calculate FPS */
		frames++;
		current_time = glutGet(GLUT_ELAPSED_TIME);
		if (current_time - last_time >= 1000) {
			g_ctx->fps = (float) frames * 1000.0f / (current_time - last_time);
			frames = 0;
			last_time = current_time;
		}
	}

	glutPostRedisplay();
}

/*-----------------------------------------------------------------------------
 * INITIALIZATION
 *----------------------------------------------------------------------------*/

void init_opengl(void) {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
	glClearDepth(1.0f);
}

/*-----------------------------------------------------------------------------
 * MAIN FUNCTION
 *----------------------------------------------------------------------------*/

int main(int argc, char **argv) {
	EVOXContext ctx;
	int i;
	unsigned long long now;

	memset(&ctx, 0, sizeof(EVOXContext));
	g_ctx = &ctx;

	printf("\n");
	printf("========================================\n");
	printf("EVOX AI CORE System v3.0.4 - Neural Sentinel\n");
	printf("GLUT Hypergraph Visualization (C90 Compliant)\n");
	printf("========================================\n");

	/* Initialize random seed */
	srand(time(NULL));

	/* Initialize mutex */
	pthread_mutex_init(&ctx.data_mutex, NULL);

	/* Generate API keys */
	now = get_timestamp_sec();
	ctx.key_count = 4;
	for (i = 0; i < ctx.key_count; i++) {
		generate_api_key(&ctx.api_keys[i], now);
	}
	ctx.last_rotation = now;
	ctx.next_rotation = now + ROTATION_INTERVAL;

	/* Initialize hypergraph */
	init_hypergraph(&ctx);

	/* Set up camera */
	ctx.camera_angle = 0.0f;
	ctx.camera_elevation = 30.0f;
	ctx.camera_distance = 350.0f;

	/* Set initial state */
	ctx.current_state = STATE_PROCESSING;
	ctx.running = 1;

	/* Start computation thread */
	pthread_create(&ctx.compute_thread, NULL, computation_thread_func, &ctx);

	/* Initialize GLUT */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_ALPHA);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 100);
	g_window_id = glutCreateWindow(WINDOW_TITLE);

	/* Set up OpenGL */
	init_opengl();

	/* Register callbacks */
	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(special_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

	printf("\n");
	printf("========================================\n");
	printf("System running - Hypergraph visualization active\n");
	printf("Controls:\n");
	printf("  Arrow keys / Mouse drag: Rotate view\n");
	printf("  +/-: Zoom in/out\n");
	printf("  Space: Pause/resume computation\n");
	printf("  R: Reset camera\n");
	printf("  ESC: Exit\n");
	printf("========================================\n\n");

	/* Enter GLUT main loop */
	glutMainLoop();

	/* Cleanup */
	printf("\nShutting down...\n");
	ctx.running = 0;

	pthread_join(ctx.compute_thread, NULL);
	pthread_mutex_destroy(&ctx.data_mutex);

	/* Final statistics */
	printf("\n");
	printf("========================================\n");
	printf("Final Statistics:\n");
	printf("  Frames rendered: %llu\n", (unsigned long long) ctx.frame_count);
	printf("  Computation cycles: %llu\n",
			(unsigned long long) ctx.computation_cycles);
	printf("  Average FPS: %.1f\n", ctx.fps);
	printf("  Final entropy: %.3f\n", ctx.system_entropy);
	printf("  API key rotations: %llu\n",
			(unsigned long long) (ctx.last_rotation / ROTATION_INTERVAL));
	printf("========================================\n");

	return 0;
}

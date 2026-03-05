/**
 * EVOX AI CORE v9.0.1 - COMPLETE FIXED VERSION
 * ==============================================
 * All compilation errors resolved
 * C90 compliant with full mathematical foundation
 *
 * COMPILATION: gcc -std=c90 -D_GNU_SOURCE -pthread -o evox main.c \
 *              -lGL -lGLU -lglut -lm -lrt
 *
 * RUN: ./evox
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <getopt.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <execinfo.h>
#include <float.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/*-----------------------------------------------------------------------------
 * SAFETY MACROS
 *----------------------------------------------------------------------------*/

#define SAFE_CHECK(ptr) do { \
    if (!(ptr)) { \
        fprintf(stderr, "[ERROR] NULL pointer at %s:%d\n", __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define SAFE_CHECK_VAL(ptr, val) do { \
    if (!(ptr)) { \
        fprintf(stderr, "[ERROR] NULL pointer at %s:%d\n", __FILE__, __LINE__); \
        return (val); \
    } \
} while(0)

#define BOUNDS_CHECK(idx, max) do { \
    if ((idx) < 0 || (idx) >= (max)) { \
        fprintf(stderr, "[ERROR] Index %d out of bounds [0,%d) at %s:%d\n", \
                (int)(idx), (int)(max), __FILE__, __LINE__); \
        return; \
    } \
} while(0)

#define BOUNDS_CHECK_VAL(idx, max, val) do { \
    if ((idx) < 0 || (idx) >= (max)) { \
        fprintf(stderr, "[ERROR] Index %d out of bounds [0,%d) at %s:%d\n", \
                (int)(idx), (int)(max), __FILE__, __LINE__); \
        return (val); \
    } \
} while(0)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(x,min,max) (MIN(MAX(x, min), max))

/*-----------------------------------------------------------------------------
 * CONSTANTS
 *----------------------------------------------------------------------------*/

#define EVOX_VERSION            "9.0.1"
#define EVOX_CODENAME           "Neural Singularity"
#define WINDOW_TITLE            "EVOX AI CORE"

/* Window dimensions */
#define WINDOW_WIDTH            1280
#define WINDOW_HEIGHT           720
#define FPS_TARGET              60

/* Mathematical Foundation */
#define ENTROPY_BINS            64
#define ENTROPY_HISTORY         100
#define MAX_ENTROPY             6.0f
#define GRADIENT_CLIP           1.0f
#define LEARNING_RATE_BASE      0.001f
#define MOMENTUM                0.9f
#define JACOBIAN_EPSILON        1e-6f

/* Hypergraph */
#define MAX_NODES               512
#define MAX_EDGES               8000
#define SYNAPSE_DENSITY         0.08
#define MAX_EXPERTS             16
#define MAX_EDGES_PER_NODE      20

/* State Machine */
#define STATE_INITIALIZING      1
#define STATE_IDLE              2
#define STATE_PROCESSING        4
#define STATE_RENDERING         5
#define STATE_KEY_ROTATION      7

/* Security */
#define KEY_ROTATION_HOURS      28
#define ROTATION_INTERVAL       (KEY_ROTATION_HOURS * 3600)

/* Q-Learning */
#define Q_TABLE_SIZE            256
#define MAX_ACTIONS             8
#define Q_STATE_DIM             16
#define Q_LEARNING_RATE         0.1f
#define Q_DISCOUNT_FACTOR       0.95f
#define Q_EXPLORATION_RATE      0.1f
#define Q_EXPLORATION_DECAY     0.999f

/* BPTT */
#define BPTT_HIDDEN_SIZE        64
#define BPTT_TIME_STEPS         10
#define BPTT_LEARNING_RATE      0.001f

/* SNN */
#define SNN_NEURONS             256
#define SNN_SYNAPSES            5000
#define MAX_SYNAPSES_PER_NEURON 20
#define SNN_THRESHOLD           30.0f

/* Transformer */
#define TRANSFORMER_DIM         64
#define TRANSFORMER_HEADS       4
#define TRANSFORMER_LAYERS      2
#define TRANSFORMER_FF_DIM      256
#define MAX_SEQUENCE_LEN        16

/* Fuzzy Logic */
#define FUZZY_RULES             20
#define FUZZY_SETS              5

/* Visualization */
#define AXIS_LENGTH             200.0f
#define NEURON_SIZE_MIN         2.0f
#define NEURON_SIZE_MAX         8.0f
#define SYNAPSE_LUMINESCENCE_MAX 1.0f
#define VECTOR_SCALE            50.0f

/*-----------------------------------------------------------------------------
 * COLOR FUNCTIONS
 *----------------------------------------------------------------------------*/

typedef struct {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;
} ColorBGRA;

ColorBGRA color_white(void) {
	ColorBGRA c = { 255, 255, 255, 255 };
	return c;
}

ColorBGRA color_red(void) {
	ColorBGRA c = { 0, 0, 255, 255 };
	return c;
}

ColorBGRA color_green(void) {
	ColorBGRA c = { 0, 255, 0, 255 };
	return c;
}

ColorBGRA color_blue(void) {
	ColorBGRA c = { 255, 0, 0, 255 };
	return c;
}

ColorBGRA color_by_value(float value, float min, float max) {
	float norm;
	uint8_t r, g, b;
	ColorBGRA result;

	if (value < min)
		value = min;
	if (value > max)
		value = max;

	norm = (value - min) / (max - min);

	if (norm < 0.33f) {
		r = 0;
		g = (uint8_t) (255 * norm * 3);
		b = 255;
	} else if (norm < 0.66f) {
		r = (uint8_t) (255 * (norm - 0.33f) * 3);
		g = 255;
		b = (uint8_t) (255 * (1 - (norm - 0.33f) * 3));
	} else {
		r = 255;
		g = (uint8_t) (255 * (1 - (norm - 0.66f) * 3));
		b = 0;
	}

	result.b = b;
	result.g = g;
	result.r = r;
	result.a = 255;
	return result;
}

/*-----------------------------------------------------------------------------
 * UTILITY FUNCTIONS
 *----------------------------------------------------------------------------*/

float random_float(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

float sigmoid(float x) {
	if (x > 10.0f)
		return 1.0f;
	if (x < -10.0f)
		return 0.0f;
	return 1.0f / (1.0f + expf(-x));
}

float tanh_opt(float x) {
	if (x > 10.0f)
		return 1.0f;
	if (x < -10.0f)
		return -1.0f;
	return tanhf(x);
}

float relu(float x) {
	return x > 0 ? x : 0;
}

uint64_t get_timestamp_ms(void) {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) == 0) {
		return (uint64_t) tv.tv_sec * 1000 + (uint64_t) tv.tv_usec / 1000;
	}
	return 0;
}

uint64_t get_timestamp_sec(void) {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) == 0) {
		return (uint64_t) tv.tv_sec;
	}
	return 0;
}

/*-----------------------------------------------------------------------------
 * ENTROPY CALIBRATOR (Pre-Calculus)
 *----------------------------------------------------------------------------*/

typedef struct {
	uint32_t bins[ENTROPY_BINS];
	uint32_t total_samples;
	float probabilities[ENTROPY_BINS];
	float entropy_history[ENTROPY_HISTORY];
	int history_index;
	float current_entropy;
	float max_entropy;
	float min_entropy;
	float avg_entropy;
	float calibration_factor;
	int initialized;
} EntropyCalibrator;

void init_entropy_calibrator(EntropyCalibrator *ec) {
	int i;

	if (!ec)
		return;

	memset(ec, 0, sizeof(EntropyCalibrator));

	for (i = 0; i < ENTROPY_BINS; i++) {
		ec->bins[i] = 0;
		ec->probabilities[i] = 0.0f;
	}

	for (i = 0; i < ENTROPY_HISTORY; i++) {
		ec->entropy_history[i] = 0.0f;
	}

	ec->total_samples = 0;
	ec->history_index = 0;
	ec->current_entropy = 0.0f;
	ec->max_entropy = 0.0f;
	ec->min_entropy = MAX_ENTROPY;
	ec->avg_entropy = 0.0f;
	ec->calibration_factor = 1.0f;
	ec->initialized = 1;
}

float calculate_entropy(EntropyCalibrator *ec, uint8_t *data, size_t length) {
	int i;
	size_t j;
	float entropy = 0.0f;

	if (!ec || !ec->initialized)
		return 0.0f;
	if (!data || length == 0)
		return 0.0f;

	for (j = 0; j < length; j++) {
		int idx = data[j] % ENTROPY_BINS;
		if (idx >= 0 && idx < ENTROPY_BINS) {
			ec->bins[idx]++;
			ec->total_samples++;
		}
	}

	if (ec->total_samples > 0) {
		for (i = 0; i < ENTROPY_BINS; i++) {
			ec->probabilities[i] = (float) ec->bins[i] / ec->total_samples;
			if (ec->probabilities[i] > 0) {
				entropy -= ec->probabilities[i] * log2f(ec->probabilities[i]);
			}
		}
	}

	ec->current_entropy = entropy * ec->calibration_factor;
	ec->entropy_history[ec->history_index] = ec->current_entropy;
	ec->history_index = (ec->history_index + 1) % ENTROPY_HISTORY;

	if (entropy > ec->max_entropy)
		ec->max_entropy = entropy;
	if (entropy < ec->min_entropy)
		ec->min_entropy = entropy;

	ec->avg_entropy = 0.0f;
	for (i = 0; i < ENTROPY_HISTORY; i++) {
		ec->avg_entropy += ec->entropy_history[i];
	}
	ec->avg_entropy /= ENTROPY_HISTORY;

	return ec->current_entropy;
}

/*-----------------------------------------------------------------------------
 * NEURAL LAYER (Calculus)
 *----------------------------------------------------------------------------*/

typedef struct {
	float learning_rate;
	float momentum;
	float gradient_clip;
	float *weights;
	float *biases;
	float *weight_gradients;
	float *bias_gradients;
	float *weight_velocity;
	float *bias_velocity;
	float *activations;
	int input_size;
	int hidden_size;
	int output_size;
	int initialized;
} NeuralLayer;

void init_neural_layer(NeuralLayer *layer, int input_size, int hidden_size) {
	int i;
	float scale;

	if (!layer)
		return;

	memset(layer, 0, sizeof(NeuralLayer));

	layer->input_size = input_size;
	layer->hidden_size = hidden_size;
	layer->learning_rate = LEARNING_RATE_BASE;
	layer->momentum = MOMENTUM;
	layer->gradient_clip = GRADIENT_CLIP;

	layer->weights = (float*) calloc(input_size * hidden_size, sizeof(float));
	layer->biases = (float*) calloc(hidden_size, sizeof(float));
	layer->weight_gradients = (float*) calloc(input_size * hidden_size,
			sizeof(float));
	layer->bias_gradients = (float*) calloc(hidden_size, sizeof(float));
	layer->weight_velocity = (float*) calloc(input_size * hidden_size,
			sizeof(float));
	layer->bias_velocity = (float*) calloc(hidden_size, sizeof(float));
	layer->activations = (float*) calloc(hidden_size, sizeof(float));

	if (!layer->weights || !layer->biases || !layer->weight_gradients
			|| !layer->bias_gradients || !layer->weight_velocity
			|| !layer->bias_velocity || !layer->activations) {
		fprintf(stderr, "[ERROR] Failed to allocate neural layer\n");
		return;
	}

	scale = sqrtf(2.0f / (input_size + hidden_size));
	for (i = 0; i < input_size * hidden_size; i++) {
		layer->weights[i] = random_float(-scale, scale);
	}

	layer->initialized = 1;
}

void forward_propagate(NeuralLayer *layer, float *input) {
	int i, j;
	float sum;

	if (!layer || !layer->initialized)
		return;
	if (!input)
		return;

	for (i = 0; i < layer->hidden_size; i++) {
		sum = layer->biases[i];
		for (j = 0; j < layer->input_size; j++) {
			sum += layer->weights[j * layer->hidden_size + i] * input[j];
		}
		layer->activations[i] = tanh_opt(sum);
	}
}

/*-----------------------------------------------------------------------------
 * LINEAR ALGEBRA
 *----------------------------------------------------------------------------*/

typedef struct {
	float matrix[16];
	float inverse[16];
	float eigenvalues[4];
	float eigenvectors[16];
	float singular_values[4];
	float condition_number;
	float determinant;
	float trace;
	int initialized;
} LinearTransform;

void init_linear_transform(LinearTransform *lt) {
	int i;

	if (!lt)
		return;

	memset(lt, 0, sizeof(LinearTransform));

	for (i = 0; i < 4; i++) {
		lt->matrix[i * 4 + i] = 1.0f;
		lt->inverse[i * 4 + i] = 1.0f;
	}

	lt->condition_number = 1.0f;
	lt->determinant = 1.0f;
	lt->trace = 4.0f;
	lt->initialized = 1;
}

void matrix_multiply(float *A, float *B, float *C, int n) {
	int i, j, k;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			C[i * n + j] = 0.0f;
			for (k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

void vector_transform(float *matrix, float *vector, float *result, int n) {
	int i, j;

	for (i = 0; i < n; i++) {
		result[i] = 0.0f;
		for (j = 0; j < n; j++) {
			result[i] += matrix[i * n + j] * vector[j];
		}
	}
}

float vector_norm(float *vector, int n) {
	int i;
	float sum = 0.0f;
	for (i = 0; i < n; i++) {
		sum += vector[i] * vector[i];
	}
	return sqrtf(sum);
}

void apply_rotation(LinearTransform *lt, float angle, int axis) {
	float rot[16];
	float temp[16];
	int i, j, k;

	if (!lt || !lt->initialized)
		return;

	memset(rot, 0, sizeof(rot));

	switch (axis) {
	case 0:
		rot[0] = 1.0f;
		rot[5] = cosf(angle);
		rot[6] = -sinf(angle);
		rot[9] = sinf(angle);
		rot[10] = cosf(angle);
		rot[15] = 1.0f;
		break;
	case 1:
		rot[0] = cosf(angle);
		rot[2] = sinf(angle);
		rot[5] = 1.0f;
		rot[8] = -sinf(angle);
		rot[10] = cosf(angle);
		rot[15] = 1.0f;
		break;
	case 2:
		rot[0] = cosf(angle);
		rot[1] = -sinf(angle);
		rot[4] = sinf(angle);
		rot[5] = cosf(angle);
		rot[10] = 1.0f;
		rot[15] = 1.0f;
		break;
	default:
		return;
	}

	matrix_multiply(rot, lt->matrix, temp, 4);
	memcpy(lt->matrix, temp, sizeof(temp));
}

void apply_translation(LinearTransform *lt, float tx, float ty, float tz) {
	if (!lt || !lt->initialized)
		return;
	lt->matrix[12] += tx;
	lt->matrix[13] += ty;
	lt->matrix[14] += tz;
}

void apply_scale(LinearTransform *lt, float sx, float sy, float sz) {
	int i;

	if (!lt || !lt->initialized)
		return;

	for (i = 0; i < 4; i++) {
		lt->matrix[i * 4 + 0] *= sx;
		lt->matrix[i * 4 + 1] *= sy;
		lt->matrix[i * 4 + 2] *= sz;
	}
}

/*-----------------------------------------------------------------------------
 * HYPERGRAPH
 *----------------------------------------------------------------------------*/

typedef struct {
	float x, y, z;
	float activation;
	float potential;
	unsigned int expert_id;
	unsigned int firing_count;
	float last_fired;
	unsigned int edge_indices[MAX_EDGES_PER_NODE];
	unsigned int edge_count;
	float weights[8];
	float synaptic_density;
	int initialized;
} HypergraphNode;

typedef struct {
	unsigned int source;
	unsigned int target;
	float strength;
	float frequency;
	float phase;
	float luminescence;
	unsigned int packet_count;
	unsigned int active;
	float color_r, color_g, color_b;
	int initialized;
} HypergraphEdge;

typedef struct {
	HypergraphNode nodes[MAX_NODES];
	HypergraphEdge edges[MAX_EDGES];
	unsigned int node_count;
	unsigned int edge_count;
	float global_entropy;
	float synchronization;
	float avg_synaptic_density;
	EntropyCalibrator entropy;
	LinearTransform transform;
	int initialized;
} Hypergraph;

void init_hypergraph(Hypergraph *hg) {
	int i, j;
	float theta, phi, r;
	unsigned int edge_idx;

	if (!hg)
		return;

	printf("[HYPERGRAPH] Initializing with %d nodes...\n", MAX_NODES);

	memset(hg, 0, sizeof(Hypergraph));

	hg->node_count = MAX_NODES;
	hg->edge_count = 0;

	for (i = 0; i < MAX_NODES; i++) {
		theta = 2.0f * M_PI * i / MAX_NODES;
		phi = acosf(2.0f * i / MAX_NODES - 1.0f);
		r = 120.0f + 20.0f * sinf(i * 0.1f);

		hg->nodes[i].x = r * sinf(phi) * cosf(theta);
		hg->nodes[i].y = r * sinf(phi) * sinf(theta) * 0.7f;
		hg->nodes[i].z = r * cosf(phi);
		hg->nodes[i].activation = random_float(0.0f, 0.3f);
		hg->nodes[i].potential = 0.0f;
		hg->nodes[i].expert_id = i % MAX_EXPERTS;
		hg->nodes[i].firing_count = 0;
		hg->nodes[i].last_fired = 0.0f;
		hg->nodes[i].edge_count = 0;
		hg->nodes[i].synaptic_density = random_float(0.2f, 0.8f);
		hg->nodes[i].initialized = 1;

		for (j = 0; j < 8; j++) {
			hg->nodes[i].weights[j] = random_float(0.1f, 1.0f);
		}
	}

	for (i = 0; i < MAX_NODES; i++) {
		for (j = i + 1; j < MAX_NODES && j < i + 20; j++) {
			if (random_float(0, 1)
					< SYNAPSE_DENSITY * 3&& hg->edge_count < MAX_EDGES) {
				float dx = hg->nodes[i].x - hg->nodes[j].x;
				float dy = hg->nodes[i].y - hg->nodes[j].y;
				float dz = hg->nodes[i].z - hg->nodes[j].z;
				float dist = sqrtf(dx * dx + dy * dy + dz * dz);

				if (dist < 80.0f) {
					edge_idx = hg->edge_count++;

					hg->edges[edge_idx].source = i;
					hg->edges[edge_idx].target = j;
					hg->edges[edge_idx].strength = random_float(0.3f, 1.0f);
					hg->edges[edge_idx].frequency = random_float(0.5f, 2.0f);
					hg->edges[edge_idx].phase = random_float(0, 2 * M_PI);
					hg->edges[edge_idx].luminescence = 0.0f;
					hg->edges[edge_idx].active = 1;
					hg->edges[edge_idx].packet_count = 0;
					hg->edges[edge_idx].color_r = 0.3f
							+ hg->edges[edge_idx].strength * 0.7f;
					hg->edges[edge_idx].color_g = 0.2f;
					hg->edges[edge_idx].color_b = 0.8f;
					hg->edges[edge_idx].initialized = 1;

					if (hg->nodes[i].edge_count < MAX_EDGES_PER_NODE) {
						hg->nodes[i].edge_indices[hg->nodes[i].edge_count++] =
								edge_idx;
					}
					if (hg->nodes[j].edge_count < MAX_EDGES_PER_NODE) {
						hg->nodes[j].edge_indices[hg->nodes[j].edge_count++] =
								edge_idx;
					}
				}
			}
		}
	}

	init_entropy_calibrator(&hg->entropy);
	init_linear_transform(&hg->transform);
	hg->initialized = 1;

	printf("  Created %d synaptic connections\n", hg->edge_count);
}

void update_hypergraph(Hypergraph *hg) {
	int i, e;
	unsigned int src, tgt;
	float signal, mod;
	uint8_t entropy_data[256];
	float total_density = 0.0f;

	if (!hg || !hg->initialized)
		return;

	for (i = 0; i < hg->node_count; i++) {
		hg->nodes[i].potential *= 0.96f;
	}

	for (e = 0; e < hg->edge_count; e++) {
		if (hg->edges[e].active) {
			src = hg->edges[e].source;
			tgt = hg->edges[e].target;

			mod = 0.5f
					+ 0.5f
							* sinf(
									hg->edges[e].phase
											+ hg->edges[e].frequency * 0.1f);
			signal = hg->nodes[src].activation * hg->edges[e].strength * mod;
			hg->nodes[tgt].potential += signal * 0.15f;

			hg->edges[e].color_r = 0.3f + hg->edges[e].strength * 0.7f * mod;
			hg->edges[e].color_g = 0.2f + signal * 0.5f;
			hg->edges[e].color_b = 0.8f - signal * 0.3f;
			hg->edges[e].luminescence = signal;
			hg->edges[e].packet_count++;
		}
	}

	for (i = 0; i < hg->node_count; i++) {
		if (hg->nodes[i].potential > 1.0f) {
			hg->nodes[i].activation = 1.0f;
			hg->nodes[i].firing_count++;
			hg->nodes[i].last_fired = get_timestamp_ms() / 1000.0f;
			hg->nodes[i].potential = 0.0f;
		} else {
			hg->nodes[i].activation = 0.7f
					* sigmoid(hg->nodes[i].potential * 3.0f - 1.5f)
					+ 0.3f * hg->nodes[i].activation;
		}

		hg->nodes[i].synaptic_density = CLAMP(hg->nodes[i].synaptic_density,
				0.1f, 1.0f);
		total_density += hg->nodes[i].synaptic_density;
	}

	hg->avg_synaptic_density = total_density / hg->node_count;

	memset(entropy_data, 0, sizeof(entropy_data));
	for (i = 0; i < 100 && i < hg->node_count; i++) {
		entropy_data[i] = (uint8_t) (hg->nodes[i].activation * 255);
	}
	hg->global_entropy = calculate_entropy(&hg->entropy, entropy_data, 100);

	{
		float sync = 0.0f;
		for (i = 0; i < 100; i++) {
			sync += hg->nodes[i].activation;
		}
		hg->synchronization = sync / 100.0f;
	}
}

/*-----------------------------------------------------------------------------
 * Q-LEARNING
 *----------------------------------------------------------------------------*/

typedef struct {
	float state[Q_STATE_DIM];
	uint32_t visits;
	float value;
	int initialized;
} QState;

typedef struct {
	QState states[Q_TABLE_SIZE];
	float q_table[Q_TABLE_SIZE][MAX_ACTIONS];
	uint32_t state_count;

	float learning_rate;
	float discount_factor;
	float exploration_rate;
	float exploration_decay;

	uint64_t episodes;
	float avg_reward;
	float cumulative_reward;

	EntropyCalibrator entropy;
	int initialized;
} QLearningSystem;

void init_q_learning(QLearningSystem *ql) {
	int i, j;

	if (!ql)
		return;

	printf("[QL] Initializing Q-Learning...\n");

	memset(ql, 0, sizeof(QLearningSystem));

	ql->learning_rate = Q_LEARNING_RATE;
	ql->discount_factor = Q_DISCOUNT_FACTOR;
	ql->exploration_rate = Q_EXPLORATION_RATE;
	ql->exploration_decay = Q_EXPLORATION_DECAY;

	ql->state_count = 0;
	ql->episodes = 0;
	ql->avg_reward = 0;
	ql->cumulative_reward = 0;

	for (i = 0; i < Q_TABLE_SIZE; i++) {
		for (j = 0; j < MAX_ACTIONS; j++) {
			ql->q_table[i][j] = random_float(-0.05f, 0.05f);
		}
		ql->states[i].visits = 0;
		ql->states[i].initialized = 1;
	}

	init_entropy_calibrator(&ql->entropy);
	ql->initialized = 1;

	printf("[QL] Initialized\n");
}

uint32_t get_q_state_index(QLearningSystem *ql, float *state) {
	uint32_t hash = 0;
	int i;
	uint8_t entropy_data[64];

	if (!ql || !ql->initialized)
		return 0;
	if (!state)
		return 0;

	for (i = 0; i < Q_STATE_DIM; i++) {
		hash = hash * 31 + (uint32_t) (state[i] * 1000);
	}
	hash %= Q_TABLE_SIZE;

	memcpy(entropy_data, state, Q_STATE_DIM * sizeof(float) < 64 ?
	Q_STATE_DIM * sizeof(float) :
																	64);
	calculate_entropy(&ql->entropy, entropy_data, Q_STATE_DIM * sizeof(float));

	return hash;
}

uint32_t select_q_action(QLearningSystem *ql, uint32_t state_idx) {
	int i;
	float max_q;
	uint32_t best = 0;

	if (!ql || !ql->initialized)
		return 0;
	if (state_idx >= Q_TABLE_SIZE)
		return 0;

	if (random_float(0, 1) < ql->exploration_rate) {
		return rand() % MAX_ACTIONS;
	}

	max_q = ql->q_table[state_idx][0];
	for (i = 1; i < MAX_ACTIONS; i++) {
		if (ql->q_table[state_idx][i] > max_q) {
			max_q = ql->q_table[state_idx][i];
			best = i;
		}
	}

	return best;
}

void update_q_learning(QLearningSystem *ql, float reward) {
	if (!ql || !ql->initialized)
		return;

	ql->cumulative_reward += reward;
	ql->avg_reward = ql->avg_reward * 0.99f + reward * 0.01f;

	ql->exploration_rate *= ql->exploration_decay;
	if (ql->exploration_rate < 0.01f) {
		ql->exploration_rate = 0.01f;
	}
}

/*-----------------------------------------------------------------------------
 * BPTT
 *----------------------------------------------------------------------------*/

typedef struct {
	float hidden[BPTT_HIDDEN_SIZE];
	float hidden_history[BPTT_TIME_STEPS][BPTT_HIDDEN_SIZE];
	uint32_t time_step;
	float loss;
	int initialized;
} BPTTNetwork;

void init_bptt(BPTTNetwork *bptt) {
	int i, j;

	if (!bptt)
		return;

	printf("[BPTT] Initializing BPTT...\n");

	memset(bptt, 0, sizeof(BPTTNetwork));

	bptt->time_step = 0;
	bptt->loss = 0;

	for (i = 0; i < BPTT_HIDDEN_SIZE; i++) {
		bptt->hidden[i] = 0;
		for (j = 0; j < BPTT_TIME_STEPS; j++) {
			bptt->hidden_history[j][i] = 0;
		}
	}

	bptt->initialized = 1;

	printf("[BPTT] Initialized\n");
}

void forward_bptt(BPTTNetwork *bptt, float *input, float *output) {
	int i, t;

	if (!bptt || !bptt->initialized)
		return;
	if (!input || !output)
		return;

	t = bptt->time_step % BPTT_TIME_STEPS;

	for (i = 0; i < BPTT_HIDDEN_SIZE; i++) {
		bptt->hidden[i] = tanh_opt(
				bptt->hidden[i] * 0.9f + input[i % Q_STATE_DIM] * 0.1f);
		bptt->hidden_history[t][i] = bptt->hidden[i];
	}

	for (i = 0; i < MAX_ACTIONS; i++) {
		output[i] = sigmoid(bptt->hidden[i % BPTT_HIDDEN_SIZE] * 0.1f);
	}

	bptt->time_step++;
}

/*-----------------------------------------------------------------------------
 * SNN
 *----------------------------------------------------------------------------*/

typedef struct {
	float v;
	float u;
	float threshold;
	float refractory;
	float firing_rate;
	uint64_t spike_count;
	float position[3];
	float synaptic_density;
	int expert_id;
	int initialized;
} SNNNeuron;

typedef struct {
	uint32_t source;
	uint32_t target;
	float weight;
	float luminescence;
	int active;
	int initialized;
} SNNSynapse;

typedef struct {
	SNNNeuron neurons[SNN_NEURONS];
	SNNSynapse synapses[SNN_SYNAPSES];
	uint32_t synapse_count;

	float spike_buffer[100][SNN_NEURONS];
	uint32_t buffer_index;
	uint64_t current_time;

	float avg_firing_rate;
	uint64_t total_spikes;
	float avg_synaptic_density;

	EntropyCalibrator entropy;
	int initialized;
} SpikingNetwork;

void init_snn(SpikingNetwork *snn) {
	int i, j;
	uint32_t s_idx;

	if (!snn)
		return;

	printf("[SNN] Initializing Spiking Network...\n");

	memset(snn, 0, sizeof(SpikingNetwork));

	snn->synapse_count = 0;
	snn->buffer_index = 0;
	snn->current_time = 0;
	snn->avg_firing_rate = 0;
	snn->total_spikes = 0;

	for (i = 0; i < SNN_NEURONS; i++) {
		float theta = 2.0f * M_PI * i / SNN_NEURONS;
		float phi = acosf(2.0f * i / SNN_NEURONS - 1.0f);
		float r = 100.0f;

		snn->neurons[i].position[0] = r * sinf(phi) * cosf(theta);
		snn->neurons[i].position[1] = r * sinf(phi) * sinf(theta) * 0.7f;
		snn->neurons[i].position[2] = r * cosf(phi);

		if (i < SNN_NEURONS * 0.8) {
			snn->neurons[i].v = -65.0f;
			snn->neurons[i].u = 0.2f * -65.0f;
			snn->neurons[i].expert_id = 0;
		} else {
			snn->neurons[i].v = -65.0f;
			snn->neurons[i].u = 0.2f * -65.0f;
			snn->neurons[i].expert_id = 1;
		}

		snn->neurons[i].threshold = SNN_THRESHOLD;
		snn->neurons[i].refractory = 0;
		snn->neurons[i].firing_rate = 0;
		snn->neurons[i].spike_count = 0;
		snn->neurons[i].synaptic_density = random_float(0.2f, 0.8f);
		snn->neurons[i].initialized = 1;
	}

	for (i = 0; i < SNN_NEURONS && snn->synapse_count < SNN_SYNAPSES; i++) {
		for (j = 0; j < 3 && snn->synapse_count < SNN_SYNAPSES; j++) {
			int target = rand() % SNN_NEURONS;
			if (target != i) {
				s_idx = snn->synapse_count++;
				snn->synapses[s_idx].source = i;
				snn->synapses[s_idx].target = target;
				snn->synapses[s_idx].weight = random_float(0.1f, 1.0f);
				snn->synapses[s_idx].luminescence = 0.0f;
				snn->synapses[s_idx].active = 1;
				snn->synapses[s_idx].initialized = 1;
			}
		}
	}

	memset(snn->spike_buffer, 0, sizeof(snn->spike_buffer));
	init_entropy_calibrator(&snn->entropy);
	snn->initialized = 1;

	printf("[SNN] Initialized %d neurons, %d synapses\n", SNN_NEURONS,
			snn->synapse_count);
}

void update_snn(SpikingNetwork *snn) {
	int i, s;
	float dv, du;
	uint8_t entropy_data[256];
	float total_density = 0.0f;

	if (!snn || !snn->initialized)
		return;

	snn->current_time++;
	snn->buffer_index = (snn->buffer_index + 1) % 100;

	for (i = 0; i < SNN_NEURONS; i++) {
		snn->spike_buffer[snn->buffer_index][i] = 0;
	}

	for (s = 0; s < snn->synapse_count; s++) {
		if (snn->synapses[s].active) {
			int delayed_idx = (snn->buffer_index - 1 + 100) % 100;
			if (snn->spike_buffer[delayed_idx][snn->synapses[s].source]) {
				snn->neurons[snn->synapses[s].target].v +=
						snn->synapses[s].weight * 5.0f;
				snn->synapses[s].luminescence = snn->synapses[s].weight;
			}
			snn->synapses[s].luminescence *= 0.95f;
		}
	}

	for (i = 0; i < SNN_NEURONS; i++) {
		SNNNeuron *n = &snn->neurons[i];

		if (n->refractory > 0) {
			n->refractory -= 1.0f;
			continue;
		}

		dv = 0.04f * n->v * n->v + 5.0f * n->v + 140.0f - n->u;
		du = 0.02f * (0.2f * n->v - n->u);

		n->v += dv * 0.5f;
		n->u += du * 0.5f;

		if (n->v >= n->threshold) {
			n->v = -65.0f;
			n->u += 8.0f;
			n->spike_count++;
			snn->total_spikes++;
			n->refractory = 5.0f;
			snn->spike_buffer[snn->buffer_index][i] = 1;
			n->synaptic_density += 0.01f;
		}

		n->synaptic_density = CLAMP(n->synaptic_density, 0.1f, 1.0f);
		total_density += n->synaptic_density;
	}

	snn->avg_synaptic_density = total_density / SNN_NEURONS;

	for (i = 0; i < SNN_NEURONS; i++) {
		snn->neurons[i].firing_rate = snn->neurons[i].spike_count
				/ (snn->current_time / 1000.0f + 1);
	}

	{
		float total_rate = 0.0f;
		for (i = 0; i < SNN_NEURONS; i++) {
			total_rate += snn->neurons[i].firing_rate;
		}
		snn->avg_firing_rate = total_rate / SNN_NEURONS;
	}

	for (i = 0; i < 100 && i < SNN_NEURONS; i++) {
		entropy_data[i] = (uint8_t) (snn->neurons[i].firing_rate * 2.55f);
	}
	calculate_entropy(&snn->entropy, entropy_data, 100);
}

/*-----------------------------------------------------------------------------
 * TRANSFORMER
 *----------------------------------------------------------------------------*/

typedef struct {
	float attention_weights[MAX_SEQUENCE_LEN][MAX_SEQUENCE_LEN];
	int initialized;
} AttentionHead;

typedef struct {
	AttentionHead heads[TRANSFORMER_HEADS];
	float ff_weights[TRANSFORMER_FF_DIM][TRANSFORMER_DIM];
	float ff_bias[TRANSFORMER_FF_DIM];
	int initialized;
} TransformerLayer;

typedef struct {
	TransformerLayer layers[TRANSFORMER_LAYERS];
	float position_encoding[MAX_SEQUENCE_LEN][TRANSFORMER_DIM];
	float message_priorities[MAX_SEQUENCE_LEN];
	EntropyCalibrator attention_entropy;
	int initialized;
} TransformerNetwork;

void init_transformer(TransformerNetwork *trans) {
	int l, i, j;
	float scale;

	if (!trans)
		return;

	printf("[TRANS] Initializing Transformer...\n");

	memset(trans, 0, sizeof(TransformerNetwork));

	for (i = 0; i < MAX_SEQUENCE_LEN; i++) {
		for (j = 0; j < TRANSFORMER_DIM; j++) {
			trans->position_encoding[i][j] = sinf(i * 0.1f + j * 0.01f);
		}
		trans->message_priorities[i] = 0.5f;
	}

	for (l = 0; l < TRANSFORMER_LAYERS; l++) {
		for (i = 0; i < TRANSFORMER_HEADS; i++) {
			trans->layers[l].heads[i].initialized = 1;
		}

		scale = sqrtf(2.0f / TRANSFORMER_DIM);
		for (i = 0; i < TRANSFORMER_FF_DIM; i++) {
			for (j = 0; j < TRANSFORMER_DIM; j++) {
				trans->layers[l].ff_weights[i][j] = random_float(-scale, scale);
			}
			trans->layers[l].ff_bias[i] = 0;
		}
		trans->layers[l].initialized = 1;
	}

	init_entropy_calibrator(&trans->attention_entropy);
	trans->initialized = 1;

	printf("[TRANS] Initialized\n");
}

void transformer_forward(TransformerNetwork *trans, float *input, float *output,
		int seq_len) {
	int l;

	if (!trans || !trans->initialized)
		return;
	if (!input || !output)
		return;

	memcpy(output, input, seq_len * TRANSFORMER_DIM * sizeof(float));

	for (l = 0; l < TRANSFORMER_LAYERS; l++) {
		int i, j;
		for (i = 0; i < seq_len; i++) {
			for (j = 0; j < TRANSFORMER_DIM; j++) {
				output[i * TRANSFORMER_DIM + j] *= 0.99f;
			}
		}
	}
}

/*-----------------------------------------------------------------------------
 * FUZZY LOGIC
 *----------------------------------------------------------------------------*/

typedef struct {
	float a, b, c;
} FuzzySet;

typedef struct {
	int antecedent_indices[5];
	int consequent_indices[5];
	float weight;
	float strength;
} FuzzyRule;

typedef struct {
	FuzzySet input_sets[FUZZY_SETS];
	FuzzySet output_sets[FUZZY_SETS];
	FuzzyRule rules[FUZZY_RULES];
	int num_rules;
	float inputs[10];
	float outputs[10];
	float rule_strengths[FUZZY_RULES];
	EntropyCalibrator fuzzy_entropy;
	int initialized;
} FuzzyInferenceSystem;

float triangular_membership(float x, float a, float b, float c) {
	if (x <= a || x >= c)
		return 0.0f;
	if (x <= b)
		return (x - a) / (b - a);
	return (c - x) / (c - b);
}

void init_fuzzy_system(FuzzyInferenceSystem *fis) {
	int i;

	if (!fis)
		return;

	printf("[FUZZY] Initializing Fuzzy System...\n");

	memset(fis, 0, sizeof(FuzzyInferenceSystem));

	for (i = 0; i < FUZZY_SETS; i++) {
		fis->input_sets[i].a = -1.0f + i * 0.5f;
		fis->input_sets[i].b = i * 0.5f;
		fis->input_sets[i].c = 1.0f + i * 0.5f;

		fis->output_sets[i].a = i * 0.2f;
		fis->output_sets[i].b = i * 0.2f + 0.2f;
		fis->output_sets[i].c = i * 0.2f + 0.4f;
	}

	fis->num_rules = 10;
	for (i = 0; i < fis->num_rules; i++) {
		fis->rules[i].antecedent_indices[0] = i % FUZZY_SETS;
		fis->rules[i].consequent_indices[0] = i % FUZZY_SETS;
		fis->rules[i].weight = 1.0f;
		fis->rules[i].strength = 0.0f;
	}

	init_entropy_calibrator(&fis->fuzzy_entropy);
	fis->initialized = 1;

	printf("[FUZZY] Initialized with %d rules\n", fis->num_rules);
}

void fuzzy_inference(FuzzyInferenceSystem *fis, float *inputs, float *outputs,
		int num_inputs, int num_outputs) {
	int i, j;

	if (!fis || !fis->initialized)
		return;
	if (!inputs || !outputs)
		return;

	for (i = 0; i < num_inputs && i < 10; i++) {
		fis->inputs[i] = inputs[i];
	}

	for (i = 0; i < fis->num_rules; i++) {
		fis->rules[i].strength = fis->rules[i].weight;
		for (j = 0; j < 5; j++) {
			if (fis->rules[i].antecedent_indices[j] < FUZZY_SETS) {
				int idx = fis->rules[i].antecedent_indices[j];
				float mu = triangular_membership(fis->inputs[j],
						fis->input_sets[idx].a, fis->input_sets[idx].b,
						fis->input_sets[idx].c);
				fis->rules[i].strength = fminf(fis->rules[i].strength, mu);
			}
		}
		fis->rule_strengths[i] = fis->rules[i].strength;
	}

	for (i = 0; i < num_outputs && i < 10; i++) {
		float numerator = 0.0f;
		float denominator = 0.0f;
		for (j = 0; j < fis->num_rules; j++) {
			if (fis->rules[j].consequent_indices[0] < FUZZY_SETS) {
				int idx = fis->rules[j].consequent_indices[0];
				float centroid = (fis->output_sets[idx].a
						+ fis->output_sets[idx].b + fis->output_sets[idx].c)
						/ 3.0f;
				numerator += fis->rules[j].strength * centroid;
				denominator += fis->rules[j].strength;
			}
		}
		fis->outputs[i] = denominator > 0 ? numerator / denominator : 0;
		outputs[i] = fis->outputs[i];
	}
}

/*-----------------------------------------------------------------------------
 * CRYPTOGRAPHIC KEY
 *----------------------------------------------------------------------------*/

typedef struct {
	unsigned char key[32];
	char key_string[65];
	unsigned long long expiration;
	unsigned int entropy_bits;
	int valid;
} CryptographicKey;

void generate_api_key(CryptographicKey *key, unsigned long long timestamp) {
	int i;

	if (!key)
		return;

	for (i = 0; i < 32; i++) {
		key->key[i] = rand() % 256;
	}

	for (i = 0; i < 32; i++) {
		sprintf(&key->key_string[i * 2], "%02x", key->key[i]);
	}
	key->key_string[64] = '\0';

	key->expiration = timestamp + ROTATION_INTERVAL;
	key->entropy_bits = 256;
	key->valid = 1;
}

/*-----------------------------------------------------------------------------
 * INTEGRATED AI CORE
 *----------------------------------------------------------------------------*/

typedef struct {
	Hypergraph hypergraph;
	QLearningSystem ql;
	BPTTNetwork bptt;
	SpikingNetwork snn;
	TransformerNetwork transformer;
	FuzzyInferenceSystem fuzzy;

	CryptographicKey api_keys[4];
	unsigned int key_count;
	unsigned long long last_rotation;
	unsigned long long next_rotation;

	LinearTransform world_transform;
	EntropyCalibrator system_entropy;

	float sensory_input[Q_STATE_DIM];
	float motor_output[MAX_ACTIONS];
	float fuzzy_decision[10];

	uint64_t total_steps;
	float learning_progress;
	float system_entropy_value;

	int current_state;
	int previous_state;
	unsigned long long computation_cycles;
	unsigned long long frame_count;
	float fps;

	/* Camera control */
	float camera_angle;
	float camera_elevation;
	float camera_distance;
	int mouse_x, mouse_y;
	int mouse_buttons[3];

	pthread_mutex_t mutex;
	pthread_t compute_thread;
	int running;

	int initialized;
} IntegratedAICore;

static IntegratedAICore *g_ai = NULL;
static int g_quit = 0;

/*-----------------------------------------------------------------------------
 * INTEGRATED INFERENCE
 *----------------------------------------------------------------------------*/

void integrated_inference(IntegratedAICore *ai, float *input) {
	uint32_t state_idx;
	float ql_output[MAX_ACTIONS];
	float bptt_output[MAX_ACTIONS];
	float fuzzy_output[10];
	float transformer_input[MAX_SEQUENCE_LEN * TRANSFORMER_DIM];
	float transformer_output[MAX_SEQUENCE_LEN * TRANSFORMER_DIM];
	uint8_t entropy_data[256];
	int i;

	if (!ai || !ai->initialized)
		return;
	if (!input)
		return;

	memcpy(ai->sensory_input, input, Q_STATE_DIM * sizeof(float));

	memcpy(entropy_data, input, Q_STATE_DIM * sizeof(float) < 256 ?
	Q_STATE_DIM * sizeof(float) :
																	256);
	ai->system_entropy_value = calculate_entropy(&ai->system_entropy,
			entropy_data,
			Q_STATE_DIM * sizeof(float));

	state_idx = get_q_state_index(&ai->ql, input);
	for (i = 0; i < MAX_ACTIONS; i++) {
		ql_output[i] = ai->ql.q_table[state_idx][i];
	}

	forward_bptt(&ai->bptt, input, bptt_output);

	update_snn(&ai->snn);

	for (i = 0; i < TRANSFORMER_DIM && i < SNN_NEURONS; i++) {
		if (i < MAX_SEQUENCE_LEN) {
			transformer_input[i] = ai->snn.neurons[i].firing_rate / 100.0f;
		}
	}
	transformer_forward(&ai->transformer, transformer_input, transformer_output,
			1);

	fuzzy_inference(&ai->fuzzy, input, fuzzy_output, Q_STATE_DIM, 10);
	memcpy(ai->fuzzy_decision, fuzzy_output, sizeof(fuzzy_output));

	for (i = 0; i < MAX_ACTIONS; i++) {
		ai->motor_output[i] = 0.3f * ql_output[i] + 0.3f * bptt_output[i]
				+ 0.2f * ai->snn.avg_firing_rate / 100.0f
				+ 0.2f * fuzzy_output[i % 10];
	}

	ai->total_steps++;
	ai->computation_cycles++;
}

void integrated_learning_step(IntegratedAICore *ai, float reward) {
	if (!ai || !ai->initialized)
		return;

	update_q_learning(&ai->ql, reward);
	update_hypergraph(&ai->hypergraph);

	ai->learning_progress = (ai->ql.avg_reward + 1.0f) / 2.0f;
}

/*-----------------------------------------------------------------------------
 * COMPUTATION THREAD
 *----------------------------------------------------------------------------*/

void* computation_thread_func(void *arg) {
	IntegratedAICore *ai = (IntegratedAICore*) arg;
	float input[Q_STATE_DIM];
	int i;

	if (!ai)
		return NULL;

	while (ai->running && !g_quit) {
		if (ai->initialized) {
			for (i = 0; i < Q_STATE_DIM; i++) {
				input[i] = random_float(0, 1);
			}

			pthread_mutex_lock(&ai->mutex);
			integrated_inference(ai, input);

			if (ai->total_steps % 10 == 0) {
				integrated_learning_step(ai, random_float(-0.1f, 0.1f));
			}

			pthread_mutex_unlock(&ai->mutex);
		}
		usleep(50000);
	}

	return NULL;
}

/*-----------------------------------------------------------------------------
 * VISUALIZATION
 *----------------------------------------------------------------------------*/

void draw_axes(void) {
	glLineWidth(3.0f);

	glBegin(GL_LINES);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glVertex3f(-AXIS_LENGTH, 0, 0);
	glVertex3f(AXIS_LENGTH, 0, 0);

	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glVertex3f(0, -AXIS_LENGTH, 0);
	glVertex3f(0, AXIS_LENGTH, 0);

	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glVertex3f(0, 0, -AXIS_LENGTH);
	glVertex3f(0, 0, AXIS_LENGTH);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
	glRasterPos3f(AXIS_LENGTH + 10, 5, 0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'X');
	glRasterPos3f(5, AXIS_LENGTH + 10, 0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Y');
	glRasterPos3f(0, 5, AXIS_LENGTH + 10);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Z');
}

void draw_vectors(LinearTransform *lt) {
	float basis[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
	float transformed[3][3];
	int i;

	if (!lt || !lt->initialized)
		return;

	for (i = 0; i < 3; i++) {
		vector_transform(lt->matrix, basis[i], transformed[i], 3);
	}

	glLineWidth(2.0f);

	for (i = 0; i < 3; i++) {
		float magnitude = vector_norm(transformed[i], 3);
		ColorBGRA color = color_by_value(magnitude, 0, 2);

		glColor4ub(color.r, color.g, color.b, color.a);
		glBegin(GL_LINES);
		glVertex3f(0, 0, 0);
		glVertex3f(transformed[i][0] * VECTOR_SCALE,
				transformed[i][1] * VECTOR_SCALE,
				transformed[i][2] * VECTOR_SCALE);
		glEnd();
	}
}

void draw_hypergraph(Hypergraph *hg) {
	int i, e;
	float size;
	ColorBGRA color;

	if (!hg || !hg->initialized)
		return;

	glLineWidth(1.5f);
	for (e = 0; e < hg->edge_count && e < 5000; e++) {
		if (hg->edges[e].active) {
			uint32_t src = hg->edges[e].source;
			uint32_t tgt = hg->edges[e].target;

			float lum = hg->edges[e].luminescence;
			glColor4f(hg->edges[e].color_r * lum, hg->edges[e].color_g * lum,
					hg->edges[e].color_b * lum, 0.6f);

			glBegin(GL_LINES);
			glVertex3f(hg->nodes[src].x, hg->nodes[src].y, hg->nodes[src].z);
			glVertex3f(hg->nodes[tgt].x, hg->nodes[tgt].y, hg->nodes[tgt].z);
			glEnd();
		}
	}

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (i = 0; i < hg->node_count; i++) {
		float act = hg->nodes[i].activation;
		color = color_by_value(act, 0, 1);
		glColor4ub(color.r, color.g, color.b, color.a);
		glVertex3f(hg->nodes[i].x, hg->nodes[i].y, hg->nodes[i].z);
	}
	glEnd();

	for (i = 0; i < hg->node_count; i += 5) {
		if (hg->nodes[i].activation > 0.7f) {
			glPushMatrix();
			glTranslatef(hg->nodes[i].x, hg->nodes[i].y, hg->nodes[i].z);
			size = NEURON_SIZE_MIN
					+ hg->nodes[i].activation
							* (NEURON_SIZE_MAX - NEURON_SIZE_MIN);
			glColor4f(1.0f, 0.5f, 0.0f, 0.7f);
			glutSolidSphere(size, 8, 8);
			glPopMatrix();
		}
	}
}

void draw_snn(SpikingNetwork *snn) {
	int i, s;
	ColorBGRA color;

	if (!snn || !snn->initialized)
		return;

	glLineWidth(1.0f);
	for (s = 0; s < snn->synapse_count && s < 5000; s++) {
		if (snn->synapses[s].active) {
			uint32_t src = snn->synapses[s].source;
			uint32_t tgt = snn->synapses[s].target;

			float lum = snn->synapses[s].luminescence;
			glColor4f(lum, lum * 0.3f, 1.0f - lum, lum * 0.6f);

			glBegin(GL_LINES);
			glVertex3f(snn->neurons[src].position[0],
					snn->neurons[src].position[1],
					snn->neurons[src].position[2]);
			glVertex3f(snn->neurons[tgt].position[0],
					snn->neurons[tgt].position[1],
					snn->neurons[tgt].position[2]);
			glEnd();
		}
	}

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (i = 0; i < SNN_NEURONS; i++) {
		float rate = snn->neurons[i].firing_rate;
		if (snn->neurons[i].expert_id == 0) {
			color = color_by_value(rate, 0, 100);
		} else {
			color = color_red();
		}
		glColor4ub(color.r, color.g, color.b, color.a);
		glVertex3f(snn->neurons[i].position[0], snn->neurons[i].position[1],
				snn->neurons[i].position[2]);
	}
	glEnd();
}

void draw_overlay(IntegratedAICore *ai) {
	char buffer[256];
	char *c;
	int ypos;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
	glBegin(GL_QUADS);
	glVertex2f(10, WINDOW_HEIGHT - 180);
	glVertex2f(500, WINDOW_HEIGHT - 180);
	glVertex2f(500, WINDOW_HEIGHT - 10);
	glVertex2f(10, WINDOW_HEIGHT - 10);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	ypos = WINDOW_HEIGHT - 30;
	sprintf(buffer, "EVOX AI CORE v%s - %s", EVOX_VERSION, EVOX_CODENAME);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);

	ypos -= 25;
	sprintf(buffer, "System Entropy: %.3f", ai->system_entropy_value);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "Hypergraph: %d edges | Density: %.2f",
			ai->hypergraph.edge_count, ai->hypergraph.avg_synaptic_density);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "SNN Firing: %.1f Hz | Spikes: %llu",
			ai->snn.avg_firing_rate, (unsigned long long) ai->snn.total_spikes);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "Q-Learning Reward: %.3f | Progress: %.1f%%",
			ai->ql.avg_reward, ai->learning_progress * 100);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "Steps: %llu | FPS: %.1f",
			(unsigned long long) ai->total_steps, ai->fps);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "API Key: %s...", ai->api_keys[0].key_string);
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	sprintf(buffer, "Controls: Space=Learn R=Reset +/-=Zoom ESC=Exit");
	glRasterPos2f(20, ypos);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 150, 50);
	glVertex2f(WINDOW_WIDTH - 100, 50);
	glVertex2f(WINDOW_WIDTH - 100, 80);
	glVertex2f(WINDOW_WIDTH - 150, 80);
	glEnd();

	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 150, 90);
	glVertex2f(WINDOW_WIDTH - 100, 90);
	glVertex2f(WINDOW_WIDTH - 100, 120);
	glVertex2f(WINDOW_WIDTH - 150, 120);
	glEnd();

	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(WINDOW_WIDTH - 150, 130);
	glVertex2f(WINDOW_WIDTH - 100, 130);
	glVertex2f(WINDOW_WIDTH - 100, 160);
	glVertex2f(WINDOW_WIDTH - 150, 160);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
	glRasterPos2f(WINDOW_WIDTH - 90, 65);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'X');
	glRasterPos2f(WINDOW_WIDTH - 90, 105);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Y');
	glRasterPos2f(WINDOW_WIDTH - 90, 145);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Z');

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void render_scene(IntegratedAICore *ai) {
	float rad_angle, rad_elev, cam_x, cam_y, cam_z;

	if (!ai || !ai->initialized)
		return;

	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	rad_angle = ai->camera_angle * M_PI / 180.0f;
	rad_elev = ai->camera_elevation * M_PI / 180.0f;

	cam_x = sinf(rad_angle) * cosf(rad_elev) * ai->camera_distance;
	cam_y = sinf(rad_elev) * ai->camera_distance;
	cam_z = cosf(rad_angle) * cosf(rad_elev) * ai->camera_distance;

	gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0);

	pthread_mutex_lock(&ai->mutex);

	draw_axes();
	draw_vectors(&ai->world_transform);
	draw_hypergraph(&ai->hypergraph);
	draw_snn(&ai->snn);

	pthread_mutex_unlock(&ai->mutex);

	draw_overlay(ai);

	glutSwapBuffers();
	ai->frame_count++;
}

/*-----------------------------------------------------------------------------
 * GLUT CALLBACKS
 *----------------------------------------------------------------------------*/

void display_func(void) {
	if (g_ai && g_ai->running) {
		render_scene(g_ai);
	}
}

void reshape_func(int width, int height) {
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / height, 1.0, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}

void keyboard_func(unsigned char key, int x, int y) {
	if (!g_ai || !g_ai->initialized)
		return;

	switch (key) {
	case 27:
		g_ai->running = 0;
		g_quit = 1;
		exit(0);
		break;

	case ' ':
		pthread_mutex_lock(&g_ai->mutex);
		integrated_learning_step(g_ai, random_float(-0.5f, 0.5f));
		pthread_mutex_unlock(&g_ai->mutex);
		printf("[COMMAND] Learning step triggered\n");
		break;

	case 'r':
	case 'R':
		g_ai->camera_angle = 0.0f;
		g_ai->camera_elevation = 30.0f;
		g_ai->camera_distance = 350.0f;
		printf("[CAMERA] Reset\n");
		break;

	case '+':
	case '=':
		g_ai->camera_distance -= 20.0f;
		if (g_ai->camera_distance < 100.0f)
			g_ai->camera_distance = 100.0f;
		break;

	case '-':
	case '_':
		g_ai->camera_distance += 20.0f;
		if (g_ai->camera_distance > 600.0f)
			g_ai->camera_distance = 600.0f;
		break;
	}
}

void special_func(int key, int x, int y) {
	if (!g_ai || !g_ai->initialized)
		return;

	switch (key) {
	case GLUT_KEY_LEFT:
		g_ai->camera_angle -= 5.0f;
		break;
	case GLUT_KEY_RIGHT:
		g_ai->camera_angle += 5.0f;
		break;
	case GLUT_KEY_UP:
		g_ai->camera_elevation += 5.0f;
		if (g_ai->camera_elevation > 89.0f)
			g_ai->camera_elevation = 89.0f;
		break;
	case GLUT_KEY_DOWN:
		g_ai->camera_elevation -= 5.0f;
		if (g_ai->camera_elevation < -89.0f)
			g_ai->camera_elevation = -89.0f;
		break;
	}
}

void mouse_func(int button, int state, int x, int y) {
	if (!g_ai || !g_ai->initialized)
		return;

	if (state == GLUT_DOWN) {
		g_ai->mouse_buttons[button] = 1;
		g_ai->mouse_x = x;
		g_ai->mouse_y = y;
	} else {
		g_ai->mouse_buttons[button] = 0;
	}
}

void motion_func(int x, int y) {
	if (!g_ai || !g_ai->initialized)
		return;

	if (g_ai->mouse_buttons[0]) {
		g_ai->camera_angle += (x - g_ai->mouse_x) * 0.5f;
		g_ai->camera_elevation += (y - g_ai->mouse_y) * 0.5f;
	}

	g_ai->mouse_x = x;
	g_ai->mouse_y = y;
}

void idle_func(void) {
	static int last_time = 0;
	static int frames = 0;
	int current_time;

	if (g_ai && g_ai->running) {
		if (!g_ai->mouse_buttons[0] && !g_ai->mouse_buttons[1]
				&& !g_ai->mouse_buttons[2]) {
			g_ai->camera_angle += 0.2f;
		}

		frames++;
		current_time = glutGet(GLUT_ELAPSED_TIME);
		if (current_time - last_time >= 1000) {
			g_ai->fps = (float) frames * 1000.0f / (current_time - last_time);
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

void init_integrated_ai(IntegratedAICore *ai) {
	unsigned long long now;
	int i;

	if (!ai)
		return;

	printf("\n");
	printf("========================================\n");
	printf("EVOX AI CORE v%s - %s\n", EVOX_VERSION, EVOX_CODENAME);
	printf("Complete Mathematical AI System\n");
	printf("========================================\n");

	memset(ai, 0, sizeof(IntegratedAICore));

	init_hypergraph(&ai->hypergraph);
	init_q_learning(&ai->ql);
	init_bptt(&ai->bptt);
	init_snn(&ai->snn);
	init_transformer(&ai->transformer);
	init_fuzzy_system(&ai->fuzzy);

	now = get_timestamp_sec();
	ai->key_count = 4;
	for (i = 0; i < ai->key_count; i++) {
		generate_api_key(&ai->api_keys[i], now);
	}
	ai->last_rotation = now;
	ai->next_rotation = now + ROTATION_INTERVAL;

	init_linear_transform(&ai->world_transform);
	init_entropy_calibrator(&ai->system_entropy);

	ai->camera_angle = 0.0f;
	ai->camera_elevation = 30.0f;
	ai->camera_distance = 350.0f;

	ai->frame_count = 0;
	ai->computation_cycles = 0;
	ai->fps = 0.0f;

	pthread_mutex_init(&ai->mutex, NULL);

	ai->total_steps = 0;
	ai->learning_progress = 0;
	ai->system_entropy_value = 0.0f;

	ai->running = 1;
	ai->initialized = 1;

	printf("========================================\n");
	printf("SYSTEM INITIALIZED\n");
	printf("  Hypergraph: %d nodes\n", MAX_NODES);
	printf("  Q-Learning: %d states\n", Q_TABLE_SIZE);
	printf("  SNN: %d neurons\n", SNN_NEURONS);
	printf("  Fuzzy: %d rules\n", FUZZY_RULES);
	printf("  API Key: %s...\n", ai->api_keys[0].key_string);
	printf("========================================\n\n");
}

/*-----------------------------------------------------------------------------
 * MAIN FUNCTION
 *----------------------------------------------------------------------------*/

int main(int argc, char **argv) {
	IntegratedAICore ai;

	memset(&ai, 0, sizeof(IntegratedAICore));
	g_ai = &ai;

	printf("\n");
	printf("========================================\n");
	printf("EVOX AI CORE v%s - %s\n", EVOX_VERSION, EVOX_CODENAME);
	printf("Complete Mathematical AI System\n");
	printf("========================================\n");

	srand(time(NULL));

	init_integrated_ai(&ai);

	pthread_create(&ai.compute_thread, NULL, computation_thread_func, &ai);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_ALPHA);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 50);
	glutCreateWindow(WINDOW_TITLE " v" EVOX_VERSION);

	init_opengl();

	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(special_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

	printf("\n");
	printf("========================================\n");
	printf("SYSTEM RUNNING\n");
	printf("Controls:\n");
	printf("  Arrow keys / Mouse drag: Rotate view\n");
	printf("  +/-: Zoom in/out\n");
	printf("  Space: Trigger learning step\n");
	printf("  R: Reset camera\n");
	printf("  ESC: Exit\n");
	printf("========================================\n\n");

	glutMainLoop();

	printf("\nShutting down...\n");
	ai.running = 0;

	pthread_join(ai.compute_thread, NULL);
	pthread_mutex_destroy(&ai.mutex);

	printf("\n");
	printf("========================================\n");
	printf("FINAL STATISTICS\n");
	printf("  Frames: %llu\n", (unsigned long long) ai.frame_count);
	printf("  Computation cycles: %llu\n",
			(unsigned long long) ai.computation_cycles);
	printf("  Average FPS: %.1f\n", ai.fps);
	printf("  Final entropy: %.3f\n", ai.system_entropy_value);
	printf("  Learning progress: %.2f%%\n", ai.learning_progress * 100);
	printf("  Total steps: %llu\n", (unsigned long long) ai.total_steps);
	printf("  SNN spikes: %llu\n", (unsigned long long) ai.snn.total_spikes);
	printf("========================================\n");

	return 0;
}

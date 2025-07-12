#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

// CUDA error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- CONFIGURATION CONSTANTS ---
const int S = 1296;
const int NUM_COEFS = 14;
const int POPULATION_SIZE = 4096;
const int NUM_GENERATIONS = 50000;
const int NUM_PARENTS = POPULATION_SIZE / 4;
// --- MODIFIED --- New parameters for adaptive mutation
constexpr float INITIAL_MUTATION_RATE = 0.15f;
constexpr float MIN_MUTATION_RATE = 0.01f;
constexpr int INITIAL_MUTATION_AMOUNT = 15;
constexpr int MIN_MUTATION_AMOUNT = 2;
constexpr int STAGNATION_THRESHOLD = 100; // generations without improvement


// =================================================================================
//  KERNEL FORWARD DECLARATIONS
// =================================================================================

// --- MODIFIED --- Kernels now take int* for populations
__global__ void evaluatePopulationKernel_Optimized(const int* __restrict__ d_all_coefs, const int* __restrict__ d_mark, const float* __restrict__ d_log2_table, uint64_t* __restrict__ d_iValid_scratch, int o_start, int num_games_in_batch, uint64_t* __restrict__ d_sum_scratch);
__global__ void finalizeResultsKernel(const uint64_t* __restrict__ d_sum_scratch, double* __restrict__ d_results, int num_elements);
__global__ void initCurandKernel(curandState* states, unsigned long long seed, int n);
// --- MODIFIED --- New and updated kernel declarations
__global__ void reproduceAndMutateKernel_Adaptive(const int* __restrict__ d_current_pop, int* __restrict__ d_next_pop, const int* __restrict__ d_parent_indices, curandState* __restrict__ states, int population_size, int num_coefs, int num_parents, float mutation_rate, int mutation_amount, int generation);
__global__ void calculateDiversityKernel(const int* __restrict__ d_population, float* __restrict__ d_diversity_scratch, int population_size, int num_coefs);


// =================================================================================
//  KERNELS
// =================================================================================

__global__ void finalizeResultsKernel(const uint64_t* __restrict__ d_sum_scratch, double* __restrict__ d_results, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    d_results[idx] = (double)d_sum_scratch[idx] / S;
}

// --- MODIFIED --- Takes int* for coefficients and converts them to float on the fly
__global__ void evaluatePopulationKernel_Optimized(const int* __restrict__ d_all_coefs, const int* __restrict__ d_mark, const float* __restrict__ d_log2_table,
    uint64_t* __restrict__ d_iValid_scratch, int o_start, int num_games_in_batch,
    uint64_t* __restrict__ d_sum_scratch)
{
    int population_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_idx >= POPULATION_SIZE) return;

    const int FIRST_GUESS_TARGET_IDX = 8;
    uint64_t* my_iValid = &d_iValid_scratch[(size_t)population_idx * S];
    uint64_t* my_sum_accumulator = &d_sum_scratch[population_idx];

    float my_coefs[NUM_COEFS];
#pragma unroll
    for (int i = 0; i < NUM_COEFS; ++i) {
        // --- KEY CHANGE: Convert from integer genotype to float phenotype ---
        my_coefs[i] = (float)d_all_coefs[population_idx * NUM_COEFS + i] / 100.0f;
    }

    float scores[S];
    int k_best_first_guess;
    bool first_guess_is_invalid;

    { // First guess calculation block
        const uint64_t initial_n = S;
        const float log2_initial_n = __ldg(&d_log2_table[initial_n]);

        for (int k = 0; k < S; k++) {
            uint64_t temp_counts[NUM_COEFS] = { 0 };
            for (int i = 0; i < initial_n; i++) temp_counts[d_mark[k * S + i]]++;

            scores[k] = 0.0f;
#pragma unroll
            for (int j = 0; j < NUM_COEFS; j++) {
                if (temp_counts[j] > 0) {
                    float p = (float)temp_counts[j] / initial_n;
                    float log2_count = __ldg(&d_log2_table[temp_counts[j]]);
                    float entropy_term = -p * (log2_count - log2_initial_n);
                    scores[k] += entropy_term * my_coefs[j];
                }
            }
        }

        float best_score = 0.0f;
        k_best_first_guess = 0;
        for (int i = 0; i < S; i++) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                k_best_first_guess = i;
            }
        }

        uint64_t k_best_pure13_count = 0;
        for (int i = 0; i < initial_n; i++) {
            if (d_mark[k_best_first_guess * S + i] == 13) k_best_pure13_count++;
        }
        if (k_best_pure13_count != 1) {
            for (int i = k_best_first_guess + 1; i < S; i++) {
                if (scores[i] == best_score) {
                    uint64_t candidate_pure13_count = 0;
                    for (int j = 0; j < initial_n; j++) {
                        if (d_mark[i * S + j] == 13) candidate_pure13_count++;
                    }
                    if (candidate_pure13_count > 0) {
                        k_best_first_guess = i;
                        break;
                    }
                }
            }
        }
    }

    first_guess_is_invalid = (k_best_first_guess != FIRST_GUESS_TARGET_IDX);

    int o_end = o_start + num_games_in_batch;
    if (o_end > S) o_end = S;

    for (int o = o_start; o < o_end; o++) {
        if (first_guess_is_invalid) {
            *my_sum_accumulator += 8;
            continue;
        }

        uint64_t n = S;
        for (int i = 0; i < S; i++) my_iValid[i] = i;
        bool solved_in_time = false;

        for (int l = 1; l < 7; l++) {
            if (n == 0) { *my_sum_accumulator += 8; solved_in_time = true; break; }

            int k_best;
            if (l == 1) {
                k_best = k_best_first_guess;
            }
            else {
                // --- Use __ldg() to load from read-only cache ---
                const float log2_n = __ldg(&d_log2_table[n]);

                for (int k = 0; k < S; k++) {
                    uint64_t temp_counts[NUM_COEFS] = { 0 };
                    for (int i = 0; i < n; i++) {
                        temp_counts[d_mark[k * S + my_iValid[i]]]++;
                    }
                    scores[k] = 0.0f;
#pragma unroll
                    for (int j = 0; j < NUM_COEFS; j++) {
                        if (temp_counts[j] > 0) {
                            float p = (float)temp_counts[j] / n;
                            // --- Use __ldg() to load from read-only cache ---
                            float log2_count = __ldg(&d_log2_table[temp_counts[j]]);
                            float entropy_term = -p * (log2_count - log2_n);
                            scores[k] += entropy_term * my_coefs[j];
                        }
                    }
                }

                float best_score_l = 0.0f;
                k_best = 0;
                for (int i = 0; i < S; i++) {
                    if (scores[i] > best_score_l) {
                        best_score_l = scores[i];
                        k_best = i;
                    }
                }

                uint64_t current_k_best_pure13_count = 0;
                for (int i = 0; i < n; i++) {
                    if (d_mark[k_best * S + my_iValid[i]] == 13) current_k_best_pure13_count++;
                }
                if (current_k_best_pure13_count != 1) {
                    for (int i = k_best + 1; i < S; i++) {
                        if (scores[i] == best_score_l) {
                            uint64_t candidate_pure13_count = 0;
                            for (int j = 0; j < n; j++) {
                                if (d_mark[i * S + my_iValid[j]] == 13) candidate_pure13_count++;
                            }
                            if (candidate_pure13_count > 0) {
                                k_best = i;
                                break;
                            }
                        }
                    }
                }
            }

            int m = d_mark[k_best * S + o];
            if (m == 13) { *my_sum_accumulator += l; solved_in_time = true; break; }

            uint64_t p = n;
            n = 0;
            for (int i = 0; i < p; i++) {
                if (d_mark[my_iValid[i] * S + k_best] == m) {
                    my_iValid[n++] = my_iValid[i];
                }
            }

            if (n == 1) { *my_sum_accumulator += (l + 1); solved_in_time = true; break; }
        }

        if (!solved_in_time) {
            *my_sum_accumulator += 8;
        }
    }
}

__global__ void initCurandKernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// --- MODIFIED --- New kernel for diversity calculation
__global__ void calculateDiversityKernel(const int* __restrict__ d_population,
    float* __restrict__ d_diversity_scratch,
    int population_size, int num_coefs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    float total_distance = 0.0f;
    for (int i = 0; i < population_size; ++i) {
        if (i == idx) continue;

        float distance = 0.0f;
        for (int j = 0; j < num_coefs; ++j) {
            int diff = d_population[idx * num_coefs + j] - d_population[i * num_coefs + j];
            distance += diff * diff;
        }
        total_distance += sqrtf(distance);
    }

    d_diversity_scratch[idx] = total_distance / (population_size - 1);
}


// --- MODIFIED --- New enhanced reproduction and mutation kernel
__global__ void reproduceAndMutateKernel_Adaptive(
    const int* __restrict__ d_current_pop,
    int* __restrict__ d_next_pop,
    const int* __restrict__ d_parent_indices,
    curandState* __restrict__ states,
    int population_size,
    int num_coefs,
    int num_parents,
    float mutation_rate,
    int mutation_amount,
    int generation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= population_size) return;

    // 1. ELITISM: Keep more of the best individuals
    if (idx < num_parents / 2) {  // Increased elite retention
        int parent_idx = d_parent_indices[idx];
#pragma unroll
        for (int i = 0; i < num_coefs; ++i) {
            d_next_pop[idx * num_coefs + i] = d_current_pop[parent_idx * num_coefs + i];
        }
        return;
    }

    curandState localState = states[idx];

    // 2. IMPROVED PARENT SELECTION: Tournament selection
    int tournament_size = 3;
    int p1_idx = 0, p2_idx = 0;

    // Tournament selection for parent 1
    for (int t = 0; t < tournament_size; ++t) {
        int candidate = curand(&localState) % num_parents;
        if (t == 0 || candidate < p1_idx) p1_idx = candidate;
    }

    // Tournament selection for parent 2
    for (int t = 0; t < tournament_size; ++t) {
        int candidate = curand(&localState) % num_parents;
        if (t == 0 || candidate < p2_idx) p2_idx = candidate;
    }

    int p1_actual_idx = d_parent_indices[p1_idx];
    int p2_actual_idx = d_parent_indices[p2_idx];

    // 3. UNIFORM CROSSOVER instead of single-point
#pragma unroll
    for (int i = 0; i < num_coefs; ++i) {
        if (curand_uniform(&localState) < 0.5f) {
            d_next_pop[idx * num_coefs + i] = d_current_pop[p1_actual_idx * num_coefs + i];
        }
        else {
            d_next_pop[idx * num_coefs + i] = d_current_pop[p2_actual_idx * num_coefs + i];
        }
    }

    // 4. ADAPTIVE MUTATION with multiple strategies
#pragma unroll
    for (int i = 0; i < num_coefs; ++i) {
        if (curand_uniform(&localState) < mutation_rate) {
            float strategy = curand_uniform(&localState);

            if (strategy < 0.7f) {
                // Standard mutation
                int mutation = (curand(&localState) % (2 * mutation_amount + 1)) - mutation_amount;
                d_next_pop[idx * num_coefs + i] += mutation;
            }
            else if (strategy < 0.85f) {
                // Gaussian mutation for fine-tuning
                float gaussian = curand_normal(&localState);
                int mutation = (int)(gaussian * mutation_amount * 0.5f);
                d_next_pop[idx * num_coefs + i] += mutation;
            }
            else {
                // Large random jump for exploration
                d_next_pop[idx * num_coefs + i] = 1 + (curand(&localState) % 300);
            }

            // Clamp to valid range
            d_next_pop[idx * num_coefs + i] = min(300, max(1, d_next_pop[idx * num_coefs + i]));
        }
    }

    states[idx] = localState;
}


// =================================================================================
//  HOST CODE
// =================================================================================
int main() {
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
    printf("--- GPU-Accelerated Mastermind Optimizer ---\n");
    printf("Setting up...\n");

    std::vector<int> h_mark_flat(S * S);
    uint64_t a[5] = { 0 }, b[5] = { 0 };

    const uint64_t h_Valids[S] = {
        1111,1112,1113,1114,1115,1116,1121,1122,1123,1124,1125,1126,1131,1132,1133,1134,1135,1136,1141,1142,1143,1144,1145,1146,1151,1152,1153,1154,1155,1156,1161,1162,1163,1164,1165,1166,1211,1212,1213,1214,1215,1216,1221,1222,1223,1224,1225,1226,1231,1232,1233,1234,1235,1236,1241,1242,1243,1244,1245,1246,1251,1252,1253,1254,1255,1256,1261,1262,1263,1264,1265,1266,1311,1312,1313,1314,1315,1316,1321,1322,1323,1324,1325,1326,1331,1332,1333,1334,1335,1336,1341,1342,1343,1344,1345,1346,1351,1352,1353,1354,1355,1356,1361,1362,1363,1364,1365,1366,1411,1412,1413,1414,1415,1416,1421,1422,1423,1424,1425,1426,1431,1432,1433,1434,1435,1436,1441,1442,1443,1444,1445,1446,1451,1452,1453,1454,1455,1456,1461,1462,1463,1464,1465,1466,1511,1512,1513,1514,1515,1516,1521,1522,1523,1524,1525,1526,1531,1532,1533,1534,1535,1536,1541,1542,1543,1544,1545,1546,1551,1552,1553,1554,1555,1556,1561,1562,1563,1564,1565,1566,1611,1612,1613,1614,1615,1616,1621,1622,1623,1624,1625,1626,1631,1632,1633,1634,1635,1636,1641,1642,1643,1644,1645,1646,1651,1652,1653,1654,1655,1656,1661,1662,1663,1664,1665,1666,
        2111,2112,2113,2114,2115,2116,2121,2122,2123,2124,2125,2126,2131,2132,2133,2134,2135,2136,2141,2142,2143,2144,2145,2146,2151,2152,2153,2154,2155,2156,2161,2162,2163,2164,2165,2166,2211,2212,2213,2214,2215,2216,2221,2222,2223,2224,2225,2226,2231,2232,2233,2234,2235,2236,2241,2242,2243,2244,2245,2246,2251,2252,2253,2254,2255,2256,2261,2262,2263,2264,2265,2266,2311,2312,2313,2314,2315,2316,2321,2322,2323,2324,2325,2326,2331,2332,2333,2334,2335,2336,2341,2342,2343,2344,2345,2346,2351,2352,2353,2354,2355,2356,2361,2362,2363,2364,2365,2366,2411,2412,2413,2414,2415,2416,2421,2422,2423,2424,2425,2426,2431,2432,2433,2434,2435,2436,2441,2442,2443,2444,2445,2446,2451,2452,2453,2454,2455,2456,2461,2462,2463,2464,2465,2466,2511,2512,2513,2514,2515,2516,2521,2522,2523,2524,2525,2526,2531,2532,2533,2534,2535,2536,2541,2542,2543,2544,2545,2546,2551,2552,2553,2554,2555,2556,2561,2562,2563,2564,2565,2566,2611,2612,2613,2614,2615,2616,2621,2622,2623,2624,2625,2626,2631,2632,2633,2634,2635,2636,2641,2642,2643,2644,2645,2646,2651,2652,2653,2654,2655,2656,2661,2662,2663,2664,2665,2666,
        3111,3112,3113,3114,3115,3116,3121,3122,3123,3124,3125,3126,3131,3132,3133,3134,3135,3136,3141,3142,3143,3144,3145,3146,3151,3152,3153,3154,3155,3156,3161,3162,3163,3164,3165,3166,3211,3212,3213,3214,3215,3216,3221,3222,3223,3224,3225,3226,3231,3232,3233,3234,3235,3236,3241,3242,3243,3244,3245,3246,3251,3252,3253,3254,3255,3256,3261,3262,3263,3264,3265,3266,3311,3312,3313,3314,3315,3316,3321,3322,3323,3324,3325,3326,3331,3332,3333,3334,3335,3336,3341,3342,3343,3344,3345,3346,3351,3352,3353,3354,3355,3356,3361,3362,3363,3364,3365,3366,3411,3412,3413,3414,3415,3416,3421,3422,3423,3424,3425,3426,3431,3432,3433,3434,3435,3436,3441,3442,3443,3444,3445,3446,3451,3452,3453,3454,3455,3456,3461,3462,3463,3464,3465,3466,3511,3512,3513,3514,3515,3516,3521,3522,3523,3524,3525,3526,3531,3532,3533,3534,3535,3536,3541,3542,3543,3544,3545,3546,3551,3552,3553,3554,3555,3556,3561,3562,3563,3564,3565,3566,3611,3612,3613,3614,3615,3616,3621,3622,3623,3624,3625,3626,3631,3632,3633,3634,3635,3636,3641,3642,3643,3644,3645,3646,3651,3652,3653,3654,3655,3656,3661,3662,3663,3664,3665,3666,
        4111,4112,4113,4114,4115,4116,4121,4122,4123,4124,4125,4126,4131,4132,4133,4134,4135,4136,4141,4142,4143,4144,4145,4146,4151,4152,4153,4154,4155,4156,4161,4162,4163,4164,4165,4166,4211,4212,4213,4214,4215,4216,4221,4222,4223,4224,4225,4226,4231,4232,4233,4234,4235,4236,4241,4242,4243,4244,4245,4246,4251,4252,4253,4254,4255,4256,4261,4262,4263,4264,4265,4266,4311,4312,4313,4314,4315,4316,4321,4322,4323,4324,4325,4326,4331,4332,4333,4334,4335,4336,4341,4342,4343,4344,4345,4346,4351,4352,4353,4354,4355,4356,4361,4362,4363,4364,4365,4366,4411,4412,4413,4414,4415,4416,4421,4422,4423,4424,4425,4426,4431,4432,4433,4434,4435,4436,4441,4442,4443,4444,4445,4446,4451,4452,4453,4454,4455,4456,4461,4462,4463,4464,4465,4466,4511,4512,4513,4514,4515,4516,4521,4522,4523,4524,4525,4526,4531,4532,4533,4534,4535,4536,4541,4542,4543,4544,4545,4546,4551,4552,4553,4554,4555,4556,4561,4562,4563,4564,4565,4566,4611,4612,4613,4614,4615,4616,4621,4622,4623,4624,4625,4626,4631,4632,4633,4634,4635,4636,4641,4642,4643,4644,4645,4646,4651,4652,4653,4654,4655,4656,4661,4662,4663,4664,4665,4666,
        5111,5112,5113,5114,5115,5116,5121,5122,5123,5124,5125,5126,5131,5132,5133,5134,5135,5136,5141,5142,5143,5144,5145,5146,5151,5152,5153,5154,5155,5156,5161,5162,5163,5164,5165,5166,5211,5212,5213,5214,5215,5216,5221,5222,5223,5224,5225,5226,5231,5232,5233,5234,5235,5236,5241,5242,5243,5244,5245,5246,5251,5252,5253,5254,5255,5256,5261,5262,5263,5264,5265,5266,5311,5312,5313,5314,5315,5316,5321,5322,5323,5324,5325,5326,5331,5332,5333,5334,5335,5336,5341,5342,5343,5344,5345,5346,5351,5352,5353,5354,5355,5356,5361,5362,5363,5364,5365,5366,5411,5412,5413,5414,5415,5416,5421,5422,5423,5424,5425,5426,5431,5432,5433,5434,5435,5436,5441,5442,5443,5444,5445,5446,5451,5452,5453,5454,5455,5456,5461,5462,5463,5464,5465,5466,5511,5512,5513,5514,5515,5516,5521,5522,5523,5524,5525,5526,5531,5532,5533,5534,5535,5536,5541,5542,5543,5544,5545,5546,5551,5552,5553,5554,5555,5556,5561,5562,5563,5564,5565,5566,5611,5612,5613,5614,5615,5616,5621,5622,5623,5624,5625,5626,5631,5632,5633,5634,5635,5636,5641,5642,5643,5644,5645,5646,5651,5652,5653,5654,5655,5656,5661,5662,5663,5664,5665,5666,
        6111,6112,6113,6114,6115,6116,6121,6122,6123,6124,6125,6126,6131,6132,6133,6134,6135,6136,6141,6142,6143,6144,6145,6146,6151,6152,6153,6154,6155,6156,6161,6162,6163,6164,6165,6166,6211,6212,6213,6214,6215,6216,6221,6222,6223,6224,6225,6226,6231,6232,6233,6234,6235,6236,6241,6242,6243,6244,6245,6246,6251,6252,6253,6254,6255,6256,6261,6262,6263,6264,6265,6266,6311,6312,6313,6314,6315,6316,6321,6322,6323,6324,6325,6326,6331,6332,6333,6334,6335,6336,6341,6342,6343,6344,6345,6346,6351,6352,6353,6354,6355,6356,6361,6362,6363,6364,6365,6366,6411,6412,6413,6414,6415,6416,6421,6422,6423,6424,6425,6426,6431,6432,6433,6434,6435,6436,6441,6442,6443,6444,6445,6446,6451,6452,6453,6454,6455,6456,6461,6462,6463,6464,6465,6466,6511,6512,6513,6514,6515,6516,6521,6522,6523,6524,6525,6526,6531,6532,6533,6534,6535,6536,6541,6542,6543,6544,6545,6546,6551,6552,6553,6554,6555,6556,6561,6562,6563,6564,6565,6566,6611,6612,6613,6614,6615,6616,6621,6622,6623,6624,6625,6626,6631,6632,6633,6634,6635,6636,6641,6642,6643,6644,6645,6646,6651,6652,6653,6654,6655,6656,6661,6662,6663,6664,6665,6666
    };

    for (int i = 0; i < S; i++) {
        for (int j = i; j < S; j++) {
            uint64_t p = 0, m = 0;
            uint64_t val_i = h_Valids[i], val_j = h_Valids[j];
            a[0] = val_i; b[0] = val_j; for (int k = 1; k < 4; k++) { a[k] = a[0] % 10; b[k] = b[0] % 10; a[0] /= 10; b[0] /= 10; } a[4] = a[0]; b[4] = b[0];
            for (int k = 1; k <= 4; k++) if (a[k] == b[k]) { p++; a[k] = 0; b[k] = 9; }
            for (int k = 1; k <= 4; k++) for (int l = 1; l <= 4; l++) if (a[k] == b[l]) { m++; b[l] = 9; break; }
            int mark_val = (int)(-0.5 * p * p + 5.5 * p + m);
            h_mark_flat[i * S + j] = mark_val; h_mark_flat[j * S + i] = mark_val;
        }
    }
    for (int i = 0; i < S; i++) h_mark_flat[i * S + i] = 13;

    printf("Pre-calculating log2 table...\n");
    std::vector<float> h_log2_table(S + 1);
    h_log2_table[0] = 0;
    for (int i = 1; i <= S; ++i) {
        h_log2_table[i] = log2f(static_cast<float>(i));
    }

    // --- GPU Memory Allocation ---
    int* d_mark; double* d_results; uint64_t* d_sum_scratch;
    uint64_t* d_iValid_scratch;
    int* d_population_A, * d_population_B; // <<< NOW INTEGER
    int* d_parent_indices; curandState* d_curand_states;
    float* d_log2_table;
    float* d_diversity_scratch; // <<< NEW

    printf("Allocating memory on GPU...\n");
    gpuErrchk(cudaMalloc(&d_mark, (size_t)S * S * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_results, (size_t)POPULATION_SIZE * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_sum_scratch, (size_t)POPULATION_SIZE * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&d_iValid_scratch, (size_t)POPULATION_SIZE * S * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&d_population_A, (size_t)POPULATION_SIZE * NUM_COEFS * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_population_B, (size_t)POPULATION_SIZE * NUM_COEFS * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_parent_indices, (size_t)NUM_PARENTS * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_curand_states, (size_t)POPULATION_SIZE * sizeof(curandState)));
    gpuErrchk(cudaMalloc(&d_log2_table, (S + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_diversity_scratch, POPULATION_SIZE * sizeof(float))); // <<< NEW

    printf("Memory allocated.\n");

    gpuErrchk(cudaMemcpy(d_mark, h_mark_flat.data(), (size_t)S * S * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log2_table, h_log2_table.data(), (S + 1) * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (POPULATION_SIZE + blockSize - 1) / blockSize;
    initCurandKernel << <gridSize, blockSize >> > (d_curand_states, time(NULL), POPULATION_SIZE);
    gpuErrchk(cudaDeviceSynchronize());

    printf("Population Size: %d, Generations: %d\n", POPULATION_SIZE, NUM_GENERATIONS);
    printf("Starting search...\n\n");

    // --- MODIFIED --- Better initialization strategy
    std::vector<int> h_population_flat(POPULATION_SIZE * NUM_COEFS);
    // --- CHANGE 1: Seed with your known best coefficients (converted to int) ---
    //const int initial_coefs[NUM_COEFS] = { 110, 107, 104, 81, 62, 102, 95, 105, 88, 92, 79, 102, 80, 193 };
    const int initial_coefs[NUM_COEFS] = { 124, 134, 135, 95, 86, 125, 133, 117, 103, 110, 113, 130, 105, 232 };
    std::mt19937 rng(std::random_device{}());

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        if (i == 0) {
            // Keep the base individual
            for (int j = 0; j < NUM_COEFS; ++j) {
                h_population_flat[i * NUM_COEFS + j] = initial_coefs[j];
            }
        }
        else if (i < POPULATION_SIZE / 4) {
            // Gaussian distribution around base
            std::normal_distribution<float> gauss_dist(0.0f, 15.0f);
            for (int j = 0; j < NUM_COEFS; ++j) {
                int val = initial_coefs[j] + (int)gauss_dist(rng);
                h_population_flat[i * NUM_COEFS + j] = std::max(1, std::min(300, val));
            }
        }
        else if (i < POPULATION_SIZE / 2) {
            // Uniform distribution in extended range
            std::uniform_int_distribution<int> extended_dist(50, 200);
            for (int j = 0; j < NUM_COEFS; ++j) {
                h_population_flat[i * NUM_COEFS + j] = extended_dist(rng);
            }
        }
        else {
            // Full random for diversity
            std::uniform_int_distribution<int> uniform_dist(1, 300);
            for (int j = 0; j < NUM_COEFS; ++j) {
                h_population_flat[i * NUM_COEFS + j] = uniform_dist(rng);
            }
        }
    }
    gpuErrchk(cudaMemcpy(d_population_A, h_population_flat.data(), (size_t)POPULATION_SIZE * NUM_COEFS * sizeof(int), cudaMemcpyHostToDevice));

    int* d_current_pop = d_population_A;
    int* d_next_pop = d_population_B;

    double best_ever_score = 999.0;
    std::vector<int> best_ever_coefs(NUM_COEFS); // Store as integer

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- MODIFIED --- Add variables for adaptive strategy
    int stagnation_counter = 0;
    float current_mutation_rate = INITIAL_MUTATION_RATE;
    int current_mutation_amount = INITIAL_MUTATION_AMOUNT;

    // --- MODIFIED --- Enhanced main evolution loop
    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        gpuErrchk(cudaMemset(d_sum_scratch, 0, (size_t)POPULATION_SIZE * sizeof(uint64_t)));

        const int BATCH_SIZE = 128;
        for (int o_start = 0; o_start < S; o_start += BATCH_SIZE) {
            evaluatePopulationKernel_Optimized << <gridSize, blockSize >> > (d_current_pop, d_mark, d_log2_table, d_iValid_scratch, o_start, BATCH_SIZE, d_sum_scratch);
        }
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        finalizeResultsKernel << <gridSize, blockSize >> > (d_sum_scratch, d_results, POPULATION_SIZE);
        gpuErrchk(cudaDeviceSynchronize());

        std::vector<double> h_results(POPULATION_SIZE);
        gpuErrchk(cudaMemcpy(h_results.data(), d_results, (size_t)POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<int> h_indices(POPULATION_SIZE);
        std::iota(h_indices.begin(), h_indices.end(), 0);
        std::sort(h_indices.begin(), h_indices.end(), [&](int a, int b) { return h_results[a] < h_results[b]; });

        double best_gen_score = h_results[h_indices[0]];

        // Track improvement
        if (best_gen_score < best_ever_score) {
            best_ever_score = best_gen_score;
            stagnation_counter = 0;
            int best_individual_idx = h_indices[0];
            gpuErrchk(cudaMemcpy(best_ever_coefs.data(), &d_current_pop[best_individual_idx * NUM_COEFS], NUM_COEFS * sizeof(int), cudaMemcpyDeviceToHost));
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            printf("T+%.2fs Gen %d: !!! NEW BEST: %.8f !!!\n", elapsed.count(), gen + 1, best_ever_score);
            printf("  Coefficients: [ ");
            for (int i = 0; i < NUM_COEFS; ++i) printf("%.2f ", (float)best_ever_coefs[i] / 100.0f);
            printf("]\n");
        }
        else {
            stagnation_counter++;
        }

        // --- CHANGE 2: Reduce frequency of the expensive diversity calculation ---
        if (gen > 0 && gen % 50 == 0) {
            calculateDiversityKernel << <gridSize, blockSize >> > (
                d_current_pop, d_diversity_scratch, POPULATION_SIZE, NUM_COEFS);
            gpuErrchk(cudaDeviceSynchronize());

            // Get average diversity
            std::vector<float> h_diversity(POPULATION_SIZE);
            gpuErrchk(cudaMemcpy(h_diversity.data(), d_diversity_scratch,
                POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            float avg_diversity = std::accumulate(h_diversity.begin(), h_diversity.end(), 0.0f) / POPULATION_SIZE;

            // Adapt mutation based on diversity and stagnation
            if (avg_diversity < 50.0f || stagnation_counter > STAGNATION_THRESHOLD) {
                current_mutation_rate = std::min(0.3f, current_mutation_rate * 1.5f);
                current_mutation_amount = std::min(25, current_mutation_amount + 3);
                printf("  [Diversity: %.2f, Stagnation: %d] Increasing mutation: rate=%.3f, amount=%d\n",
                    avg_diversity, stagnation_counter, current_mutation_rate, current_mutation_amount);
                if (stagnation_counter > STAGNATION_THRESHOLD) stagnation_counter = 0; // Reset counter after adapting
            }
            else {
                current_mutation_rate = std::max(MIN_MUTATION_RATE, current_mutation_rate * 0.98f);
                current_mutation_amount = std::max(MIN_MUTATION_AMOUNT, current_mutation_amount - 1);
            }
        }

        std::vector<int> h_parent_indices(h_indices.begin(), h_indices.begin() + NUM_PARENTS);
        gpuErrchk(cudaMemcpy(d_parent_indices, h_parent_indices.data(), (size_t)NUM_PARENTS * sizeof(int), cudaMemcpyHostToDevice));

        // Use the enhanced reproduction kernel
        reproduceAndMutateKernel_Adaptive << <gridSize, blockSize >> > (
            d_current_pop, d_next_pop, d_parent_indices, d_curand_states,
            POPULATION_SIZE, NUM_COEFS, NUM_PARENTS,
            current_mutation_rate, current_mutation_amount, gen);
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(d_current_pop, d_next_pop);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;

    printf("\n--- Optimization Finished ---\n");
    printf("Total runtime: %.2f seconds\n", total_duration.count());
    printf("Best score found: %.8f\n", best_ever_score);
    printf("Optimal coefficients found (integer representation):\n[ ");
    for (int i = 0; i < NUM_COEFS; ++i) printf("%d ", best_ever_coefs[i]);
    printf("]\n");
    printf("Optimal coefficients found (float representation):\n[ ");
    for (int i = 0; i < NUM_COEFS; ++i) printf("%.2f ", (float)best_ever_coefs[i] / 100.0f);
    printf("]\n");


    gpuErrchk(cudaFree(d_mark));
    gpuErrchk(cudaFree(d_results));
    gpuErrchk(cudaFree(d_sum_scratch));
    gpuErrchk(cudaFree(d_iValid_scratch));
    gpuErrchk(cudaFree(d_population_A));
    gpuErrchk(cudaFree(d_population_B));
    gpuErrchk(cudaFree(d_parent_indices));
    gpuErrchk(cudaFree(d_curand_states));
    gpuErrchk(cudaFree(d_log2_table));
    gpuErrchk(cudaFree(d_diversity_scratch)); // <<< NEW

    system("pause");
    return 0;
}
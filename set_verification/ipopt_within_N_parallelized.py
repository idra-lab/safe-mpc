import numpy as np
from safe_mpc.set_verification_ocp import NStepControlInvarianceOCP
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

args = parse_args()
model_name = args['system']

states_to_verify = np.load('sampled_states.npy')

# Build once BEFORE forking — this writes the .so file
params = Parameters(args, model_name, rti=False)
model = AdamModel(params)
_ = NStepControlInvarianceOCP(model)  # triggers .so compilation
del _, model, params
print("Shared library built. Starting parallel workers...")

horizon_to_test = list(np.arange(1, Parameters(args, model_name, rti=False).N + 1))


def process_state(i):
    # Now each worker loads the already-existing .so (no write race)
    params_local = Parameters(args, model_name, rti=False)
    model_local = AdamModel(params_local)
    set_OCP = NStepControlInvarianceOCP(model_local)

    x_init = states_to_verify[i]
    row = np.zeros(len(horizon_to_test), dtype=bool)

    for j, horizon in enumerate(reversed(horizon_to_test)):
        result, _ = set_OCP.solveProblem(x_init, horizon)
        row[-(j + 1)] = result

    if not row.any():
        print(f'State {i}_{x_init}: Unsafe for all horizons')

    return i, row


if __name__ == '__main__':
    
    results = np.zeros((states_to_verify.shape[0], len(horizon_to_test)), dtype=bool)

    with Pool(processes=cpu_count()) as pool:
        for i, row in tqdm(
            pool.imap_unordered(process_state, range(states_to_verify.shape[0])),
            total=states_to_verify.shape[0]
        ):
            results[i] = row

    np.save('verification_results_N_stepCIS.npy', results)
    print(f'Successful states = {np.sum(results.any(axis=1))} / {results.shape[0]}')
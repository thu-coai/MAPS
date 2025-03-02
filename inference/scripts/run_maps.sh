bash ./inference/maps_v1_step1_convert.sh
python ./inference/maps_v1_step1_refine_ps.py --model_name claude-3-5-sonnet-20240620
python ./inference/maps_v1_step1_sim.py --path inference/results/simple_circuit_eval/claude35sonnet20240620_cogagent-vqa-09-02-16-42_maps_v1_step1.json
python ./inference/maps_v1_step2_sim_aided_reason.py --model_name claude-3-5-sonnet-20240620
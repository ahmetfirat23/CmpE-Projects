import os
import sys
import subprocess


def write_to_file(filename, variable):
    str = 'open("%s", write, Stream), ' % filename
    for var in variable:
        str += 'write(Stream, "%s = "), writeln(Stream, %s), ' % (var, var)
    str += 'close(Stream)'
    return str


def convert(tests, path):
    result_path = os.path.join(path, "results")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    commands = []
    for expr, var, name in tests:
        # str = "swipl --stack_limit=4G -g 'consult(\"%s\").' " % os.path.join(path, "main.pro")
        # str = str + "-g '" + expr + ', ' + write_to_file(os.path.join(result_path, name), var) + ".' "
        # str += "-g halt"
        cmd = ("swipl",
               "--stack_limit=4G",
               "-g",
               f"consult('{os.path.join(path, 'main.pro')}'), {expr}, {write_to_file(os.path.join(result_path, name), var)}.",
               "-g",
               "halt")
        commands.append(cmd)
    return commands


def consult_scene(scene_id):
    return f'consult("{os.path.join(sys.argv[1], "scene%d.pro")}"), ' % scene_id


test_cases = [
    (consult_scene(1) +
     "state(_, Agents, _, _), get_dict(0, Agents, A), get_dict(1, Agents, B), distance(A, B, Distance)",
     ["Distance"],
     "test-1-1.txt"),
    (consult_scene(1) +
     "state(_, Agents, _, _), get_dict(2, Agents, A), get_dict(0, Agents, B), distance(A, B, Distance)",
     ["Distance"],
     "test-1-2.txt"),
    (consult_scene(2) +
     "multiverse_distance(3, 0, 0, 1, Distance)",
     ["Distance"],
     "test-2-1.txt"),
    (consult_scene(2) +
     "multiverse_distance(3, 1, 0, 2, Distance)",
     ["Distance"],
     "test-2-2.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent(581, 1, NearestAgentId, Distance)",
     ["NearestAgentId", "Distance"],
     "test-3-1.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent(581, 2, NearestAgentId, Distance)",
     ["NearestAgentId", "Distance"],
     "test-3-2.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent(1261, 5, NearestAgentId, Distance)",
     ["NearestAgentId", "Distance"],
     "test-3-3.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent_in_multiverse(1390, 7, TargetStateId, TargetAgentId, Distance)",
     ["TargetStateId", "TargetAgentId", "Distance"],
     "test-4-1.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent_in_multiverse(1209, 7, TargetStateId, TargetAgentId, Distance)",
     ["TargetStateId", "TargetAgentId", "Distance"],
     "test-4-2.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), nearest_agent_in_multiverse(1390, 2, TargetStateId, TargetAgentId, Distance)",
     ["TargetStateId", "TargetAgentId", "Distance"],
     "test-4-3.txt"),
    (consult_scene(1) +
     "num_agents_in_state(3, 'Arthur', NumWarriors, NumWizards, NumRogues)",
     ["NumWarriors", "NumWizards", "NumRogues"],
     "test-5-1.txt"),
    (consult_scene(1) +
     "num_agents_in_state(3, 'Merlin', NumWarriors, NumWizards, NumRogues)",
     ["NumWarriors", "NumWizards", "NumRogues"],
     "test-5-2.txt"),
    (consult_scene(1) +
     "difficulty_of_state(3, 'Arthur', warrior, Difficulty)",
     ["Difficulty"],
     "test-6-1.txt"),
    (consult_scene(1) +
     "difficulty_of_state(3, 'Merlin', wizard, Difficulty)",
     ["Difficulty"],
     "test-6-2.txt"),
    (consult_scene(1) +
     "difficulty_of_state(3, 'Mordred', rogue, Difficulty)",
     ["Difficulty"],
     "test-6-3.txt"),
    (consult_scene(1) +
     "difficulty_of_state(3, 'Morgana', rogue, Difficulty)",
     ["Difficulty"],
     "test-6-4.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), easiest_traversable_state(581, 1,  TargetStateId)",
     ["TargetStateId"],
     "test-7-1.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), easiest_traversable_state(581, 2,  TargetStateId)",
     ["TargetStateId"],
     "test-7-2.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), easiest_traversable_state(713, 0,  TargetStateId)",
     ["TargetStateId"],
     "test-7-3.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), easiest_traversable_state(713, 6,  TargetStateId)",
     ["TargetStateId"],
     "test-7-4.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), basic_action_policy(581, 1,  Action)",
     ["Action"],
     "test-8-1.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), basic_action_policy(581, 2,  Action)",
     ["Action"],
     "test-8-2.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), basic_action_policy(581, 3,  Action)",
     ["Action"],
     "test-8-3.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), basic_action_policy(713, 0,  Action)",
     ["Action"],
     "test-8-4.txt"),
    (consult_scene(3) +
     "set_random(seed(1)), main_loop(25), basic_action_policy(713, 6,  Action)",
     ["Action"],
     "test-8-5.txt"),
]

tests = convert(test_cases, sys.argv[1])
for i, test in enumerate(tests):
    try:
        out = subprocess.run(test, timeout=100)
        if out.returncode != 0:
            print("False", file=open(os.path.join(sys.argv[1], "results", test_cases[i][-1]), "w"))
    except subprocess.TimeoutExpired:
        print("Timeout", file=open(os.path.join(sys.argv[1], "results", test_cases[i][-1]), "w"))

import numpy as np
from datetime import datetime
import os
import json
import subprocess
import os.path
import sys
from itertools import permutations, combinations
import subprocess


def generate_all_goals_in_goal_space():
    goals = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        for f in [0, 1]:
                            for g in [0, 1]:
                                for h in [0, 1]:
                                    for i in [0, 1]:
                                        goals.append([a, b, c, d, e, f, g, h, i])

    return np.array(goals)


def generate_goals():
    # Returns expert-defined buckets
    buckets = {0: [(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), (-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
                   (-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), (1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)],

               1: [(-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), (1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
                   (1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)],

               2: [(-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0), (-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0),
                   (-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0), (-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0),
                   (1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
                   (1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0), (-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0),
                   (1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0), (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0),
                   (1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0), (1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0)],

               3: [(1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0), (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0),
                   (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0), (1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0),
                   (1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0), (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
                   (1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0), (1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0),
                   (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0)],

               4: [(-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0), (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
                   (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0), (1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0),
                   (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0), (1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0)
                   ]
               }
    return buckets


# def get_instruction():
#     buckets = generate_goals()
#
#     all_goals = generate_all_goals_in_goal_space().astype(np.float32)
#     valid_goals = []
#     for k in buckets.keys():
#         # if k < 4:
#         valid_goals += buckets[k]
#     valid_goals = np.array(valid_goals)
#     all_goals = np.array(all_goals)
#     num_goals = all_goals.shape[0]
#     all_goals_str = [str(g) for g in all_goals]
#     valid_goals_str = [str(vg) for vg in valid_goals]
#
#     # initialize dict to convert from the oracle id to goals and vice versa.
#     # oracle id is position in the all_goal array
#     g_str_to_oracle_id = dict(zip(all_goals_str, range(num_goals)))
#
#     instructions = ['Bring blocks away_from each_other',
#                     'Bring blue close_to green and red far',
#                     'Bring blue close_to red and green far',
#                     'Bring green close_to red and blue far',
#                     'Bring blue close_to red and green',
#                     'Bring green close_to red and blue',
#                     'Bring red close_to green and blue',
#                     'Bring all blocks close',
#                     'Stack blue on green and red far',
#                     'Stack green on blue and red far',
#                     'Stack blue on red and green far',
#                     'Stack red on blue and green far',
#                     'Stack green on red and blue far',
#                     'Stack red on green and blue far',
#                     'Stack blue on green and red close_from green',
#                     'Stack green on blue and red close_from blue',
#                     'Stack blue on red and green close_from red',
#                     'Stack red on blue and green close_from blue',
#                     'Stack green on red and blue close_from red',
#                     'Stack red on green and blue close_from green',
#                     'Stack blue on green and red close_from both',
#                     'Stack green on blue and red close_from both',
#                     'Stack blue on red and green close_from both',
#                     'Stack red on blue and green close_from both',
#                     'Stack green on red and blue close_from both',
#                     'Stack red on green and blue close_from both',
#                     'Stack green on red and blue',
#                     'Stack red on green and blue',
#                     'Stack blue on green and red',
#                     'Stack green on blue and blue on red',
#                     'Stack red on blue and blue on green',
#                     'Stack blue on green and green on red',
#                     'Stack red on green and green on blue',
#                     'Stack green on red and red on blue',
#                     'Stack blue on red and red on green',
#                     ]
#     words = ['stack', 'green', 'blue', 'on', 'red', 'and', 'close_from', 'both', 'far', 'close', 'all', 'bring', 'blocks', 'away_from', 'close_to']
#     length = set()
#     for s in instructions:
#         if len(s) not in length:
#             length.add(len(s.split(' ')))
#
#
#     oracle_id_to_inst = dict()
#     g_str_to_inst = dict()
#     for g_str, oracle_id in g_str_to_oracle_id.items():
#         if g_str in valid_goals_str:
#             inst = instructions[valid_goals_str.index(g_str)]
#         else:
#             inst = ' '.join(np.random.choice(words, size=np.random.choice(list(length))))
#         g_str_to_inst[g_str] = inst
#         oracle_id_to_inst[g_str] = inst
#
#     return oracle_id_to_inst, g_str_to_inst

def get_instruction():
    return ['Bring blue and green apart',
            'Bring blue and green together',
            'Bring blue and red apart',
            'Bring blue and red together',
            'Bring green and blue apart',
            'Bring green and blue together',
            'Bring green and red apart',
            'Bring green and red together',
            'Bring red and blue apart',
            'Bring red and blue together',
            'Bring red and green apart',
            'Bring red and green together',
            'Get blue and green close_from each_other',
            'Get blue and green far_from each_other',
            'Get blue and red close_from each_other',
            'Get blue and red far_from each_other',
            'Get blue close_to green',
            'Get blue close_to red',
            'Get blue far_from green',
            'Get blue far_from red',
            'Get green and blue close_from each_other',
            'Get green and blue far_from each_other',
            'Get green and red close_from each_other',
            'Get green and red far_from each_other',
            'Get green close_to blue',
            'Get green close_to red',
            'Get green far_from blue',
            'Get green far_from red',
            'Get red and blue close_from each_other',
            'Get red and blue far_from each_other',
            'Get red and green close_from each_other',
            'Get red and green far_from each_other',
            'Get red close_to blue',
            'Get red close_to green',
            'Get red far_from blue',
            'Get red far_from green',
            'Put blue above green',
            'Put blue above red',
            'Put blue and green on_the_same_plane',
            'Put blue and red on_the_same_plane',
            'Put blue below green',
            'Put blue below red',
            'Put blue close_to green',
            'Put blue close_to red',
            'Put blue far_from green',
            'Put blue far_from red',
            'Put blue on_top_of green',
            'Put blue on_top_of red',
            'Put blue under green',
            'Put blue under red',
            'Put green above blue',
            'Put green above red',
            'Put green and blue on_the_same_plane',
            'Put green and red on_the_same_plane',
            'Put green below blue',
            'Put green below red',
            'Put green close_to blue',
            'Put green close_to red',
            'Put green far_from blue',
            'Put green far_from red',
            'Put green on_top_of blue',
            'Put green on_top_of red',
            'Put green under blue',
            'Put green under red',
            'Put red above blue',
            'Put red above green',
            'Put red and blue on_the_same_plane',
            'Put red and green on_the_same_plane',
            'Put red below blue',
            'Put red below green',
            'Put red close_to blue',
            'Put red close_to green',
            'Put red far_from blue',
            'Put red far_from green',
            'Put red on_top_of blue',
            'Put red on_top_of green',
            'Put red under blue',
            'Put red under green',
            'Remove blue from green',
            'Remove blue from red',
            'Remove blue from_above green',
            'Remove blue from_above red',
            'Remove blue from_below green',
            'Remove blue from_below red',
            'Remove blue from_under green',
            'Remove blue from_under red',
            'Remove green from blue',
            'Remove green from red',
            'Remove green from_above blue',
            'Remove green from_above red',
            'Remove green from_below blue',
            'Remove green from_below red',
            'Remove green from_under blue',
            'Remove green from_under red',
            'Remove red from blue',
            'Remove red from green',
            'Remove red from_above blue',
            'Remove red from_above green',
            'Remove red from_below blue',
            'Remove red from_below green',
            'Remove red from_under blue',
            'Remove red from_under green'
            ]


def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    logdir = os.path.join(args.save_dir, '{}_{}_{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.env_name, args.architecture))
    if args.masks:
        logdir += '_masks'
    logdir += '_{}'.format(args.reward_type)
    # add commit hash : 
    # args.commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path


def get_stat_func(line='mean', err='std'):

    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError

    if err == 'std':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    elif err == 'range':
        def err_plus(a):
            return np.nanmax(a, axis=0)
        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':
        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)
        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError

    return line_f, err_minus, err_plus


class CompressPDF:
    """
    author: Pure Python
    url: https://www.purepython.org
    copyright: CC BY-NC 4.0
    Forked date: 2018-01-07 / First version MIT license -- free to use as you want, cheers.
    Original Author: Sylvain Carlioz, 6/03/2017
    Simple python wrapper script to use ghoscript function to compress PDF files.
    With this class you can compress and or fix a folder with (corrupt) PDF files.
    You can also use this class within your own scripts just do a
    import CompressPDF
    Compression levels:
        0: default
        1: prepress
        2: printer
        3: ebook
        4: screen
    Dependency: Ghostscript.
    On MacOSX install via command line `brew install ghostscript`.
    """
    def __init__(self, compress_level=0, show_info=False):
        self.compress_level = compress_level

        self.quality = {
            0: '/default',
            1: '/prepress',
            2: '/printer',
            3: '/ebook',
            4: '/screen'
        }

        self.show_compress_info = show_info

    def compress(self, file=None, new_file=None):
        """
        Function to compress PDF via Ghostscript command line interface
        :param file: old file that needs to be compressed
        :param new_file: new file that is commpressed
        :return: True or False, to do a cleanup when needed
        """
        try:
            if not os.path.isfile(file):
                print("Error: invalid path for input PDF file")
                sys.exit(1)

            # Check if file is a PDF by extension
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.pdf':
                raise Exception("Error: input file is not a PDF")
                return False

            if self.show_compress_info:
                initial_size = os.path.getsize(file)

            subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                            '-dPDFSETTINGS={}'.format(self.quality[self.compress_level]),
                            '-dNOPAUSE', '-dQUIET', '-dBATCH',
                            '-sOutputFile={}'.format(new_file),
                             file]
            )


            if self.show_compress_info:
                final_size = os.path.getsize(new_file)
                ratio = 1 - (final_size / initial_size)
                print("Compression by {0:.0%}.".format(ratio))
                print("Final file size is {0:.1f}MB".format(final_size / 1000000))

            return True
        except Exception as error:
            print('Caught this error: ' + repr(error))
        except subprocess.CalledProcessError as e:
            print("Unexpected error:".format(e.output))
            return False


def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = key
            else:
                pass
    return inverse


INSTRUCTIONS = get_instruction()

language_to_id = dict(zip(INSTRUCTIONS, range(len(INSTRUCTIONS))))
id_to_language = dict(zip(range(len(INSTRUCTIONS)), INSTRUCTIONS))


def get_graph_structure(n):
    """ Given the number of blocks (nodes), returns :
    edges: in the form [to, from]
    incoming_edges: for each node, the indexes of the incoming edges
    predicate_ids: the ids of the predicates takes for each edge """
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
    edges = list(permutations(np.arange(n), 2))
    obj_ids = np.arange(n)
    n_comb = n * (n-1) // 2

    incoming_edges = []
    for obj_id in obj_ids:
        temp = []
        for i, pair in enumerate(permutations(np.arange(n), 2)):
            if obj_id == pair[0]:
                temp.append(i)
        incoming_edges.append(temp)

    predicate_ids = []
    for pair in permutations(np.arange(n), 2):
        ids_g = [i for i in range(len(map_list))
                 if (set(map_list[i]) == set(pair) and i < n_comb)
                 or (map_list[i] == pair and i >= n_comb)]
        predicate_ids.append(ids_g)

    return edges, incoming_edges, predicate_ids


def get_idxs_per_relation(n):
    """ For each possible relation between any pair of objects, outputs the corresponding predicate indexes in the goal vector"""
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
    all_relations = list(combinations(np.arange(n), 2))
    return np.array([np.array([i for i in range(len(map_list)) if set(map_list[i]) == set(r)]) for r in all_relations])


def get_idxs_per_object(n):
    """ For each objects, outputs the predicates indexes that include the corresponding object"""
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
    obj_ids = np.arange(n)
    return np.array([np.array([i for i in range(len(map_list)) if obj_id in map_list[i]]) for obj_id in obj_ids])


def get_eval_goals(instruction, n, nb_goals=1):
    """ Given an instruction and the total number of objects on the table, outputs a corresponding semantic goal"""
    res = []
    n_blocks = n
    n_comb = n_blocks * (n_blocks - 1) // 2
    n_perm = n_blocks * (n_blocks - 1)
    goal_dim = n_comb + n_perm
    try:
        predicate, pairs = instruction.split('_')
    except ValueError:
        predicate, pairs_1, pairs_2 = instruction.split('_')
    # tower + pyramid
    if predicate == 'mixed':
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=False)
            tower_objects = objects[:2]
            pyramid_objects = objects[2:]
            for j in range(1):
                obj_ids = (tower_objects[j], tower_objects[j+1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+10)

            for j in range(1):
                obj_ids = (pyramid_objects[j], pyramid_objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                    if set((pyramid_objects[j], pyramid_objects[-1])) == set(c):
                        id.append(k)
                    if j == 2 - 2 and set((pyramid_objects[j + 1], pyramid_objects[-1])) == set(c):
                        id.append(k)
            for j in range(2):
                obj_ids = (pyramid_objects[-1], pyramid_objects[j])
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)

    # trapeze
    if predicate == 'trapeze':
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=False)
            base_objects = objects[:3]
            top_objects = objects[3:]
            for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                if set((base_objects[0], base_objects[1])) == set(c):
                    id.append(k)
                if set((base_objects[1], base_objects[2])) == set(c):
                    id.append(k)
                if set((top_objects[0], top_objects[1])) == set(c):
                    id.append(k)
                if set((top_objects[0], base_objects[0])) == set(c):
                    id.append(k)
                if set((top_objects[0], base_objects[1])) == set(c):
                    id.append(k)
                if set((top_objects[1], base_objects[1])) == set(c):
                    id.append(k)
                if set((top_objects[1], base_objects[2])) == set(c):
                    id.append(k)
            for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                if (top_objects[0], base_objects[0]) == c:
                    id.append(k + n_comb)
                if (top_objects[0], base_objects[1]) == c:
                    id.append(k + n_comb)
                if (top_objects[1], base_objects[1]) == c:
                    id.append(k + n_comb)
                if (top_objects[1], base_objects[2]) == c:
                    id.append(k + n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)
    # two towers
    if predicate == '2stacks':
        stack_size_1 = int(pairs_1)
        stack_size_2 = int(pairs_2)

        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=stack_size_1 + stack_size_2, replace=False)
            for j in range(stack_size_1 - 1):
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            for j in range(stack_size_1, stack_size_1+stack_size_2-1):
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)

    # pyramid
    if predicate == 'pyramid':
        n_base = int(pairs)-1
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=n_base+1, replace=False)
            for j in range(n_base-1):
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                    if set((objects[j], objects[-1])) == set(c):
                        id.append(k)
                    if j == n_base - 2 and set((objects[j+1], objects[-1])) == set(c):
                        id.append(k)
            for j in range(n_base):
                obj_ids = (objects[-1], objects[j])
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)

    if predicate == 'stack':
        stack_size = int(pairs)
        close_pairs = 0
    else:
        stack_size = 1
        close_pairs = int(pairs)
    # no stacks whatsoever
    if stack_size == 1:
        ids = []
        for _ in range(nb_goals):
            id = np.random.choice(np.arange(n_comb), size=close_pairs, replace=False)
            ids.append(id)
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)
    # one tower
    else:
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=stack_size, replace=False)
            for j in range(stack_size-1):
                obj_ids = (objects[j], objects[j+1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            res.append(g)
        return np.array(res)



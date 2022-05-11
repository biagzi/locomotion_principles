from os import kill
import numpy as np
import re
import zlib
from collections import defaultdict



def identity(x):
    return x


def sigmoid(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def positive_sigmoid(x):
    return (1 + sigmoid(x)) * 0.5


def rescaled_positive_sigmoid(x, x_min=0, x_max=1):
    return (x_max - x_min) * positive_sigmoid(x) + x_min


def inverted_sigmoid(x):
    return sigmoid(x) ** -1


def neg_abs(x):
    return -np.abs(x)


def neg_square(x):
    return -np.square(x)


def sqrt_abs(x):
    return np.sqrt(np.abs(x))


def neg_sqrt_abs(x):
    return -sqrt_abs(x)


def mean_abs(x):
    return np.mean(np.abs(x))


def std_abs(x):
    return np.std(np.abs(x))


def count_positive(x):
    return np.sum(np.greater(x, 0))


def count_negative(x):
    return np.sum(np.less(x, 0))


def proportion_equal_to(x, keys):
    return np.mean(count_occurrences(x, keys))


def nested_dict():
    return defaultdict(nested_dict)


def normalize(x):
    x -= np.min(x)
    x /= np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x




def compressed_size(a):
    return len(zlib.compress(a))


def bootstrap_ci(a, func, n=5000, ci=95):
    stats = func(np.random.choice(a, (n, len(a))), axis=1)
    lower = np.percentile(stats, 100-ci)
    upper = np.percentile(stats, ci)
    return lower, upper


def vox_id_from_xyz(x, y, z, size):
    return z*size[0]*size[1] + y*size[0] + x


def vox_xyz_from_id(idx, size):
    z = idx / (size[0]*size[1])
    y = (idx - z*size[0]*size[1]) / size[0]
    x = idx - z*size[0]*size[1] - y*size[0]
    return x, y, z


def convert_voxelyze_index(v_index, spacing=0.010, start_pos=0.005):
    return int(v_index/spacing - start_pos)


def resize_voxarray(a, pad=2, const=0):
    if isinstance(pad, int):
        n_pad = ((pad, pad),)*3  # (n_before, n_after) for each dimension
    else:
        n_pad = pad
    return np.pad(a, pad_width=n_pad, mode='constant', constant_values=const)


def get_outer_shell(a):
    x, y, z = a.shape
    return [a[0, :, :], a[x-1, :, :], a[:, 0, :], a[:, y-1, :], a[:, :, 0], a[:, :, z-1]]


def get_outer_shell_complements(a):
    x, y, z = a.shape
    return [a[1:, :, :], a[:x-1, :, :], a[:, 1:, :], a[:, :y-1, :], a[:, :, 1:], a[:, :, :z-1]]


def trim_voxarray(a):
    new = np.array(a)
    done = False
    while not done:
        outer_slices = get_outer_shell(new)
        inner_slices = get_outer_shell_complements(new)
        for i, o in zip(inner_slices, outer_slices):
            if np.sum(o) == 0:
                new = i
                break

        voxels_in_shell = [np.sum(s) for s in outer_slices]
        if 0 not in voxels_in_shell:
            done = True

    return new



def reorder_vxa_array(a, size):
    anew = np.empty(size)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                anew[x, y, z] = a[z, y*size[0]+x]
    return anew


def array_to_vxa(a):
    anew = np.empty((a.shape[2], a.shape[1]*a.shape[0]))
    for z in range(a.shape[2]):
        for y in range(a.shape[1]):
            for x in range(a.shape[0]):
                anew[z, y*a.shape[0]+x] = a[x, y, z]
    return anew


def xml_format(tag):
    """Ensures that tag is encapsulated inside angle brackets."""
    if tag[0] != "<":
        tag = "<" + tag
    if tag[-1:] != ">":
        tag += ">"
    return tag


def get_data_from_xml_line(line, tag, dtype=float):
    try:
        return dtype(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])
    except ValueError:
        start = line.find(">")
        end = line.find("</")
        return dtype(line[start+1:end])


def natural_sort(l, reverse):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


def find_between(string, start, end):
    start = string.index(start) + len(start)
    end = string.index(end, start)
    return string[start:end]


def replace_text_in_file(filename, replacements_dict):
    lines = []
    with open(filename) as infile:
        for line in infile:
            for original, target in replacements_dict.iteritems():
                line = line.replace(original, target)
            lines.append(line)
    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line)


def dominates(ind1, ind2, attribute_name, maximize):
    """Returns True if ind1 dominates ind2 in a shared attribute."""
    if maximize:
        return getattr(ind1, attribute_name) > getattr(ind2, attribute_name)
    else:
        return getattr(ind1, attribute_name) < getattr(ind2, attribute_name)


def count_occurrences(x, keys):
    """Count the total occurrences of any keys in x."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    active = np.zeros_like(x, dtype=np.bool)
    for a in keys:
        active = np.logical_or(active, x == a)
    return active.sum()

def count_feet(x,keys):

    """Count the feet occurrences of any keys in x."""

    x = x.astype(bool)

    return np.sum(np.sum(x, axis = 0),0)[0]

def measure_diversity(x,feet,voxels):
    """Measure the diversity occurrences of any keys in x."""

    ind_equal_feet = feet.count(np.sum(np.sum(x.astype(bool), axis = 0),0)[0])
    ind_equal_vox = voxels.count(np.sum(x.astype(bool)))
    
    return ind_equal_feet + ind_equal_vox #this number represents robots with equal num of feet and voxels, so I want to minimize it.

def measure_diversity_stronger(x,feet,voxels,MAX_FEET,MAX_VOX):
    """Measure the diversity occurrences of any keys in x."""

    x_feet = np.sum(np.sum(x.astype(bool), axis = 0),0)[0]
    x_voxels = np.sum(x.astype(bool))

    feet_diffs = 0
    voxels_diffs = 0

    for i in range(len(feet)):
        feet_diffs += np.abs(feet[i] - x_feet)/MAX_FEET #if they all have equal feet, this number = 0; if they have the max diff, this number = 1;
        voxels_diffs += np.abs(voxels[i] - x_voxels)/MAX_VOX


    return feet_diffs + voxels_diffs #(MIN of this sum = 0 (all equall) and MAX (all different)) -> In this case, I want to Maximize


def total_diversity(x,all_inds_shape):
    """Measure the diversity of a ind comparing its shape matrix with each other individual in the population."""

    ind = x.astype(bool)
    equal_voxels = 0

    for other_ind in all_inds_shape:
        equal_voxels += np.logical_and(ind,other_ind).sum() #it is not normalized, so it will have a bias to generate smaller inds, 
        #because smaller inds have a smaller equal_voxels number


    return equal_voxels #we want to minimize this number


def total_diversity_normalized(x,all_inds_shape):

    """Measure the diversity of a ind comparing its shape matrix with each other individual in the population. 
    The number of identical voxels is normalized by the total voxels of both shapes"""

    ind = x.astype(bool)
    equal_voxels = 0

    for other_ind in all_inds_shape:
        equal_voxels += float(np.logical_and(ind,other_ind).sum())/(float(ind.sum() + other_ind.sum())/2)
        #if they are totally equal, this num is 1


    return equal_voxels #we want to minimize this number


def total_diversity_normalized_similar_inds(ind,pop):

    """Measure the diversity of a ind comparing its shape matrix with each other individual in the population. 
    The number of identical voxels is normalized by the total voxels of both shapes.
    Totally equal inds (or very similar) are penalized, but x of them are not (so they can keep exploring this space)"""

    SIMILARITY_THRESHOLD = 0.9 #define the robots that will be grouped together as very similar shapes that need to be diversified
    ind_shape = ind.genotype.to_phenotype_mapping['material']['state'].astype(bool)
    penalization_by_similarity = 0
    totally_equal_inds = []

    for other_ind in pop:
        other_ind_shape = other_ind.genotype.to_phenotype_mapping['material']['state'].astype(bool)
        ind_other_ind_equal_voxels = float(np.logical_and(ind_shape,other_ind_shape).sum())/(float(ind_shape.sum() + other_ind_shape.sum())/2)
        if ind_other_ind_equal_voxels >= SIMILARITY_THRESHOLD:
            totally_equal_inds.append([other_ind.fitness,other_ind.id])

    if len(totally_equal_inds) > 2:
        totally_equal_inds = sorted(totally_equal_inds, key=lambda x: x[0],reverse=True) #sort by the biggest value of fitness, x[0]
        for pos, element in enumerate(totally_equal_inds):
            if element[1] == ind.id and pos > 1: #if the ind is not the first two with higher fitness
                penalization_by_similarity = pos #the smaller the fitness, the bigger is the penalization

    return penalization_by_similarity  #we want to minimize this number



def map_genotype_phenotype_direct_encode(this_softbot, *args, **kwargs):
    mapping = this_softbot.to_phenotype_mapping
    material = mapping["material"]
    if material["dependency_order"] is not None: 
        for dependency_name in material["dependency_order"]:
            mapping.dependencies[dependency_name]["state"] = material["state"] > 0 

    if material["dependency_order"] is not None: 
        if mapping.dependencies[dependency_name]["material_if_true"] is not None: 
            material["state"][mapping.get_dependency(dependency_name, True)] = \
                mapping.dependencies[dependency_name]["material_if_true"]
    
        if mapping.dependencies[dependency_name]["material_if_false"] is not None:
            material["state"][mapping.get_dependency(dependency_name, False)] = \
                mapping.dependencies[dependency_name]["material_if_false"]
    return make_one_shape_only(material["state"]) * material["state"]


def map_genotype_phenotype_CPPN(this_softbot, *args, **kwargs):

    mapping = this_softbot.to_phenotype_mapping
    material = mapping["material"]

    if material["dependency_order"] is not None:
        for dependency_name in material["dependency_order"]: 
            for network in this_softbot: 
                if dependency_name in network.graph.nodes(): 
                    mapping.dependencies[dependency_name]["state"] = network.graph.node[dependency_name]["state"] - np.mean(network.graph.node[dependency_name]["state"]) > 0 


    if material["dependency_order"] is not None:
        for dependency_name in reversed(material["dependency_order"]): 
            
            if mapping.dependencies[dependency_name]["material_if_true"] is not None: 
                material["state"][mapping.get_dependency(dependency_name, True)] = \
                    mapping.dependencies[dependency_name]["material_if_true"]

            if mapping.dependencies[dependency_name]["material_if_false"] is not None:
                material["state"][mapping.get_dependency(dependency_name, False)] = \
                    mapping.dependencies[dependency_name]["material_if_false"]
                    
    return make_one_shape_only(material["state"]) * material["state"]



def make_one_shape_only(output_state, mask=None):
    """Find the largest continuous arrangement of True elements after applying boolean mask.

    Avoids multiple disconnected softbots in simulation counted as a single individual.

    Parameters
    ----------
    output_state : numpy.ndarray
        Network output

    mask : bool mask
        Threshold function applied to output_state

    Returns
    -------
    part_of_ind : bool
        True if component of individual

    """
    if mask is None:
        def mask(u): return np.greater(u, 0)

    # print output_state
    # sys.exit(0)

    one_shape = np.zeros(output_state.shape, dtype=np.int32)

    
    if np.sum(mask(output_state)) < 2:
        one_shape[np.where(mask(output_state))] = 1
        return one_shape

    else:
        not_yet_checked = []
        for x in range(output_state.shape[0]):
            for y in range(output_state.shape[1]):
                for z in range(output_state.shape[2]):
                    not_yet_checked.append((x, y, z))

        largest_shape = []
        queue_to_check = []
        while len(not_yet_checked) > len(largest_shape):
            queue_to_check.append(not_yet_checked.pop(0))
            this_shape = []
            if mask(output_state[queue_to_check[0]]):
                this_shape.append(queue_to_check[0])

            while len(queue_to_check) > 0:
                this_voxel = queue_to_check.pop(0)
                x = this_voxel[0]
                y = this_voxel[1]
                z = this_voxel[2]
                for neighbor in [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]:
                    if neighbor in not_yet_checked:
                        not_yet_checked.remove(neighbor)
                        if mask(output_state[neighbor]):
                            queue_to_check.append(neighbor)
                            this_shape.append(neighbor)

            if len(this_shape) > len(largest_shape):
                largest_shape = this_shape

        for loc in largest_shape:
            one_shape[loc] = 1

        return one_shape



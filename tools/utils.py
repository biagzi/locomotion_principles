import numpy as np
import re
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



def vox_xyz_from_id(idx, size):
    """ Used in networks"""
    z = idx / (size[0]*size[1])
    y = (idx - z*size[0]*size[1]) / size[0]
    x = idx - z*size[0]*size[1] - y*size[0]
    return x, y, z


def xml_format(tag):
    """Ensures that tag is encapsulated inside angle brackets."""
    if tag[0] != "<":
        tag = "<" + tag
    if tag[-1:] != ">":
        tag += ">"
    return tag



def natural_sort(l, reverse):
    """ Used in checkpoint"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


def find_between(string, start, end):
    """Used in logging"""
    start = string.index(start) + len(start)
    end = string.index(end, start)
    return string[start:end]




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


#R: Add patch in morphologies with holes?
def add_patch(a, loc=None, mat=1):
    empty_spots_on_surface = np.equal(count_neighbors(a), 1).reshape(a.shape)  # excludes corners
    patchable = np.greater(count_neighbors(empty_spots_on_surface.astype(int)), 1).reshape(a.shape)  # 2x2 patch
    patchable = np.logical_and(patchable, empty_spots_on_surface)

    if loc is None:
        # randomly select a patchable spot on surface
        rand = np.random.rand(*a.shape)
        rand[np.logical_not(patchable)] = 0
        # choice = np.argmax(rand.flatten())
        # choice = np.unravel_index(choice, a.shape)
        sorted_locations = [np.unravel_index(r, a.shape) for r in np.argsort(rand.flatten())]

    else:
        # find patchable closest to desired location
        indices = np.array([vox_xyz_from_id(idx, a.shape) for idx in np.arange(a.size)])
        flat_patchable = np.array([patchable[x, y, z] for (x, y, z) in indices])
        distances = cdist(indices, np.array([loc]))
        distances[np.logical_not(flat_patchable)] = a.size
        # closest = np.argmin(distances)
        # choice = indices[closest]
        sorted_locations = [indices[d] for d in np.argsort(distances.flatten())]

    # print sorted_locations

    attempt = -1
    correct_topology = False

    while not correct_topology:

        attempt += 1
        choice = sorted_locations[attempt]

        neigh = np.array([choice]*6)
        neigh[0, 0] += 1
        neigh[1, 0] -= 1
        neigh[2, 1] += 1
        neigh[3, 1] -= 1
        neigh[4, 2] += 1
        neigh[5, 2] -= 1

        slots = [0]*6
        for n, (x, y, z) in enumerate(neigh):
            if a.shape[0] > x > -1 and a.shape[1] > y > -1 and a.shape[2] > z > -1 and patchable[x, y, z]:
                slots[n] = 1

        # just doing 2x2 patch which means we can't select a row of 3 vox
        if slots[0] and slots[1]:
            slots[np.random.randint(2)] = 0
        if slots[2] and slots[3]:
            slots[2 + np.random.randint(2)] = 0
        if slots[4] and slots[5]:
            slots[4 + np.random.randint(2)] = 0

        # now we should have an L shape of 3 surface voxels, so we need to fill in the open corner to get a 2x2
        # todo: if patch is positioned between two limbs, which are longer than 2 vox, we can end up with 4 vox here
        corner_neigh = np.array(choice)
        if slots[0]:
            corner_neigh[0] += 1
        if slots[1]:
            corner_neigh[0] -= 1
        if slots[2]:
            corner_neigh[1] += 1
        if slots[3]:
            corner_neigh[1] -= 1
        if slots[4]:
            corner_neigh[2] += 1
        if slots[5]:
            corner_neigh[2] -= 1

        # add these vox to the structure as "sub voxels"
        sub_vox = [choice, tuple(corner_neigh)]
        new = np.array(a)
        for (x, y, z) in sub_vox:
            new[x, y, z] = mat

        for s, (x, y, z) in zip(slots, neigh):
            if s:
                new[x, y, z] = mat
                sub_vox += [(x, y, z)]

        # make sure the patch is fully fastened to the body
        if len(sub_vox) == 2:
            continue

        # triangulate
        plane = None
        for ax in range(3):
            if sub_vox[0][ax] == sub_vox[1][ax] == sub_vox[2][ax]:
                plane = ax

        parents = [list(xyz) for xyz in list(sub_vox)]
        grandchildren = list(parents)
        test_above, test_below = list(parents[0]), list(parents[0])
        test_above[plane] += 1
        xa, ya, za = test_above
        test_below[plane] -= 1
        xb, yb, zb = test_below

        if a.shape[0] > xa > -1 and a.shape[1] > ya > -1 and a.shape[2] > za > -1 and a[xa, ya, za]:
            correct_topology = True
            parents_above = True
            # for n in range(len(parents)):
            #     parents[n][plane] += 1
            #     grandchildren[n][plane] -= 1
            #     correct_topology = True
            #     parents_above = True

        elif a.shape[0] > xb > -1 and a.shape[1] > yb > -1 and a.shape[2] > zb > -1 and a[xb, yb, zb]:
            correct_topology = True
            parents_above = False
            # for n in range(len(parents)):
            #     parents[n][plane] -= 1
            #     grandchildren[n][plane] += 1
            #     correct_topology = True
            #     parents_above = False

    # also change mat for grandchildren
    if parents_above:
        pos = -1
    else:
        pos = 1

    for (x, y, z) in grandchildren:

        xx = x
        yy = y
        zz = z

        if plane == 0:
            xx += pos
        elif plane == 1:
            yy += pos
        elif plane == 2:
            zz += pos

        try:
            new[max(xx, 0), max(yy, 0), max(zz, 0)] = mat
        except IndexError:
            pass

    # parents = [vox_id_from_xyz(x, y, z, a.shape) for (x, y, z) in parents]
    # children = [vox_id_from_xyz(x, y, z, a.shape) for (x, y, z) in sub_vox]
    # grandchildren = [vox_id_from_xyz(x, y, z, a.shape) for (x, y, z) in grandchildren]

    # height_off_ground = min(2, get_depths_of_material_from_shell(new, mat)[4])
    # if height_off_ground > 0:
    #     new = new[:, :, height_off_ground:]
    #     new = np.pad(new, pad_width=((0, 0), (0, 0), (0, height_off_ground)), mode='constant', constant_values=0)

    sub_vox_dict = dict()
    # for p, (c, gc) in zip(parents, zip(children, grandchildren)):
    #     sub_vox_dict[p] = {c: gc}

    return new, sub_vox_dict
import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc


class Node:
    def __init__(self, depth, parent, size, mask=None):
        self.depth = depth
        self.is_leaf = False
        self.parent = parent
        self.children = {}
        self.actor = None
        self.attr = None
        self.size = size
        self.mask = mask

    def train(
        self,
        training_points_global,
        training_points,
        score_nodes,
        min_leaf_size,
        max_depth,
    ):

        #
        if self.size <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.attr = training_points[0]
        else:
            self.is_leaf = False
            (self.attr, split_dict) = score_nodes(
                self.depth, training_points, training_points_global, self.mask
            )
            for (i, (outcome, trn_split)) in enumerate(split_dict.items()):

                self.children[outcome] = Node(
                    depth=self.depth + 1,
                    parent=self,
                    size=len(trn_split),
                    mask=outcome,
                )

                self.children[outcome].train(
                    training_points_global,
                    trn_split,
                    score_nodes,
                    min_leaf_size,
                    max_depth,
                )

    def get_child(self, response):
        mask = []
        self_mask = []
        for i in response:
            if i != " ":
                mask.append(i)
        if self.mask != None:
            for i in self.mask:
                if i != " ":
                    self_mask.append(i)
        for (j, i) in enumerate(mask):
            if self.mask != None and self_mask[j] != "_":
                mask[j] = self_mask[j]

        mask = " ".join(mask)
        #         print(self.children.values())
        if mask not in self.children:
            print("Unseen outcome " + mask + " -- using the default_predict routine")
            exit()
        else:
            return self.children[mask]

    def get_query(self):
        return self.attr


class Tree:
    def __init__(self, min_leaf_size=1, max_depth=15):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth

    def train(self, training_points_global, score_nodes):

        self.root = Node(depth=0, parent=None, size=len(training_points_global))
        self.root.train(
            training_points_global,
            [i for i in range(len(training_points_global))],
            score_nodes,
            self.min_leaf_size,
            self.max_depth,
        )


def score_nodes(depth, training_points, training_points_global, original_mask=None):
    if depth == 0:
        split_dict = {}
        masks = [" ".join(["_" for _ in word]) for word in training_points_global]
        for index, mask in zip(training_points, masks):
            split_dict.setdefault(mask, []).append(index)
        return ("len", split_dict)
    else:
        best_attr = None
        best_split_dict = {}
        min_entropy_value = np.inf
        for index in training_points:
            attribute_choice = training_points_global[index]

            (split_dict, entropy_value) = diff_key_vals(
                training_points, attribute_choice, training_points_global, original_mask
            )

            if entropy_value < min_entropy_value:
                min_entropy_value = entropy_value
                best_split_dict = split_dict
                best_attr = index
        return (best_attr, best_split_dict)


def diff_key_vals(
    training_points, attribute_choice, training_points_global, original_mask
):
    count_dictionary = {}
    split_dict = {}
    for index in training_points:
        word = training_points_global[index]
        curr_mask = []
        for i in original_mask:
            if i != " ":
                curr_mask.append(i)

        for j in range(min(len(attribute_choice), len(word))):
            if curr_mask[j] == "_" and word[j] == attribute_choice[j]:
                curr_mask[j] = attribute_choice[j]

        curr_mask = " ".join(curr_mask)
        if curr_mask not in split_dict:
            count_dictionary[curr_mask] = 0
            split_dict[curr_mask] = []

        count_dictionary[curr_mask] += 1
        split_dict[curr_mask].append(index)

    counts = np.array(list(count_dictionary.values()))
    if np.min(counts) <= 0:
        raise ValueError("Elements with zero or negative counts detected")

    if counts.sum() <= 1:
        print(f"Warning: {counts.sum()} elements in total.")
        return 0
    proportions = counts / counts.sum()

    entropy_value = np.sum(proportions * np.log2(counts))
    return (split_dict, entropy_value)


################################
# Non Editable Region Starting #
################################
def my_fit(words):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to train your decision tree model using the word list provided
    # Return the trained model as is -- do not compress it using pickle etc
    # Model packing or compression will cause evaluation failure

    model = Tree()
    model.train(words, score_nodes)
    return model  # Return the trained model

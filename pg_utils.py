import json
import os
import random

import numpy as np

from pg_snapshot import PgSnapshot

###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################


class PostgresDataSet:
    def __init__(self, opt):
        """
        Parameters
        ----------
        opt : argparse.Namespace
            Object from argparse parse_args().

        Attributes
        ----------
        num_sample_per_q : int
            ???
        batch_size : int
            ???
        num_q : int
            ??? Presumably TPCH
        input_func
            ???
        """
        # Inherited hardcoded constant.
        train_test_split = 0.8

        self._rng = np.random.default_rng(seed=15721)
        self.batch_size = opt.batch_size

        self.db_snapshot = PgSnapshot(opt.db_name, opt.db_user, opt.db_pass)
        self.input_func = self.db_snapshot.all_input_funcs

        # Each file in the data directory should be a .txt containing psql output that includes the
        # EXPLAIN JSON plans.
        # Each file is assumed to correspond to a unique type of query (why?).

        fnames = [fname for fname in os.listdir(opt.data_dir) if "csv" in fname]
        fnames = sorted(fnames, key=lambda fname: int(fname.split("temp")[1][:-4]))
        if len(fnames) == 0:
            fnames = sorted([fname for fname in os.listdir(opt.data_dir) if fname.endswith(".txt")])
        print(fnames)

        num_per_q = min(len(self.get_all_plans(opt.data_dir + "/" + fname)) for fname in fnames)
        self.num_per_q = num_per_q
        print(f"Using {self.num_per_q=}")
        self.num_q = len(fnames)
        self.num_sample_per_q = int(num_per_q * train_test_split)

        data = []
        all_groups, all_groups_test = [], []

        self.grp_idxes = []
        self.num_grps = [0] * self.num_q
        for i, fname in enumerate(fnames):
            # Extract all the query plans from the current file.
            query_plans = self.get_all_plans(opt.data_dir + "/" + fname)

            # ---
            # This code block is relevant to both train and test.

            # Group the query plans, i.e., hash each query plan and combine queries plans with the
            # same hash into the same group. We obtain a vector denoting each query plan's group assignment,
            # and the total number of groups.
            group_assignments, num_groups = self.grouping(query_plans)

            # TODO(WAN): For some reason, we reimplement train/test splitting. However, the rest of the code is also
            #            not robust to empty groups.
            if opt.data_shuffle_hack:
                assignments = list(zip(query_plans, group_assignments))
                random.shuffle(assignments)
                query_plans, group_assignments = zip(*assignments)

            assert len(query_plans) == len(group_assignments), "Each query plan must be assigned a group."
            assignments = zip(query_plans, group_assignments)
            # Collect the query plans above by their assigned group.
            groups = [[] for _ in range(num_groups)]
            for query_plan, group_idx in assignments:
                groups[group_idx].append(query_plan)
            # Accumulate the groups.
            all_groups += groups

            # Record the number of groups for this batch of data.
            self.num_grps[i] = num_groups

            # ---
            # This code block is relevant to train.

            # We take the first num_sample_per_q query plans for training.
            self.grp_idxes += group_assignments[: self.num_sample_per_q]
            data += query_plans[: self.num_sample_per_q]

            # ---
            # This code block is relevant to test.

            # We take the remaining query plans for testing.
            test_groups = [[] for _ in range(num_groups)]
            for j, grp_idx in enumerate(group_assignments[self.num_sample_per_q :]):
                test_groups[grp_idx].append(query_plans[self.num_sample_per_q + j])
            all_groups_test += test_groups

        self.dataset = data
        self.datasize = len(self.dataset)
        print("Number of groups per query: ", self.num_grps)

        # TODO(WAN): This was pickled separately before. Arguably we just pickle the whole class now.
        # Compute normalizing constants for each operator.
        self.mean_range_dict = self.normalize()
        print(self.mean_range_dict)

        self.test_dataset = [self.get_input(grp) for grp in all_groups_test]
        self.all_dataset = [self.get_input(grp) for grp in all_groups]

    def normalize(self):  # compute the mean and std vec of each operator
        """
        For each operator, normalize each input feature to have a mean of 0 and maximum of 1

        Returns:
        - mean_range_dict: a dictionary where the keys are the Operator Names and the values are 2-tuples (mean_vec, max_vec):
            -- mean_vec : a vector of mean values for input features of this operator
            -- max_vec  : a vector of max values for input features of this operator
        """
        feat_vec_col = {operator: [] for operator in self.db_snapshot.all_dicts}

        def parse_input(data):
            feat_vec = [self.input_func[data[0]["Node Type"]](jss) for jss in data]
            if "Plans" in data[0]:
                for i in range(len(data[0]["Plans"])):
                    parse_input([jss["Plans"][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(
                np.array(feat_vec).astype(np.float32)
            )

        for i in range(self.datasize // self.num_sample_per_q):
            try:
                if self.num_grps[i] == 1:
                    parse_input(
                        self.dataset[
                            i * self.num_sample_per_q : (i + 1) * self.num_sample_per_q
                        ]
                    )
                else:
                    groups = [[] for j in range(self.num_grps[i])]
                    offset = i * self.num_sample_per_q
                    for j, plan_dict in enumerate(
                        self.dataset[offset : offset + self.num_sample_per_q]
                    ):
                        groups[self.grp_idxes[offset + j]].append(plan_dict)
                    for grp in groups:
                        parse_input(grp)
            except:
                print("i: {}".format(i))

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return (0, 1)
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (
                    np.mean(total_vec, axis=0),
                    np.max(total_vec, axis=0) + np.finfo(np.float32).eps,
                )

        mean_range_dict = {
            operator: cmp_mean_range(feat_vec_col[operator])
            for operator in self.db_snapshot.all_dicts
        }
        return mean_range_dict

    def get_all_plans(self, fname):
        """
        Parse the data file into a list of JSON dicts.

        TODO(WAN): It might be easier to enforce this at collection time, instead of parsing a file that has
                   psql output and the JSON query plans sequentially interleaved.

        Parameters
        ----------
        fname : str
            The path to the file containing query plans.

        Returns
        -------
        query_plans : List[Dict]
            A sanitized list of dictionaries, where each dictionary represents a query plan.
        """
        jsonstrs = []
        curr = ""
        prev = None
        prevprev = None
        with open(fname, "r") as f:
            for row in f:
                if not (
                    "[" in row or "{" in row or "]" in row or "}" in row or ":" in row
                ):
                    continue
                newrow = (
                    row.replace("+", "").replace("(1 row)\n", "").strip("\n").strip(" ")
                )
                if (
                    "CREATE" not in newrow
                    and "DROP" not in newrow
                    and "Tim" != newrow[:3]
                ):
                    curr += newrow
                if prevprev is not None and "Execution Time" in prevprev:
                    jsonstrs.append(curr.strip(" ").strip("QUERY PLAN").strip("-"))
                    curr = ""
                prevprev = prev
                prev = newrow

        strings = [s for s in jsonstrs if s[-1] == "]"]
        jss = [json.loads(s)[0]["Plan"] for s in strings]
        # jss is a list of json-transformed dicts, one for each query
        return jss

    def grouping(self, query_plans):
        """
        Group the query plans.

        Let || denote concatenation. For each query plan qp,
            hash(qp) = qp["Node Type"] || (hash(child) for child in qp["Plans"])
        Query plans which have the same hash are grouped together.

        Parameters
        ----------
        query_plans : List[Dict]
            A list of dictionaries, where each dictionary represents a query plan.

        Returns
        -------
        group_assignments : List[int]
            The corresponding group for each query plan in the original data.
        num_groups : int
            The number of groups, where each group represents a distinct query plan.
        """
        # TODO(WAN): Computing hashes is embarassingly parallel with a collect operation at the end.
        def hash(plan_dict):
            res = plan_dict["Node Type"]
            if "Plans" in plan_dict:
                for chld in plan_dict["Plans"]:
                    res += hash(chld)
            return res

        counter = 0
        string_hash = []
        enum = []
        for plan_dict in query_plans:
            string = hash(plan_dict)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)
        print(f"{counter} distinct templates identified")
        print(f"Operators: {string_hash}")
        assert counter > 0, "There must be at least one query plan."
        assert len(enum) == len(
            query_plans
        ), "Each input query plan must have been assigned a group."
        return enum, counter

    def get_input(self, data):
        """
        Vectorize the input of a list of queries that have the same plan structure.

        Parameters
        ----------
        data : List[Dict]
            A list of dictionaries, each corresponding to a query plan OF IDENTICAL STRUCTURE.
            This is not checked! Watch out.

        Returns
        -------
        new_samp_dict : Dict
            A dictionary containing the following attributes:
                node_type : str             : Name of the operator.
                is_subplan : bool           : True if the queries are subplans, false otherwise.
                subbatch_size : int         : Number of queries.
                children_plan : List[Dict]  : List of dictionaries, where each dictionary is obtained by recursively
                                              invoking this function on the corresponding child of the current node,
                                              i.e., first element corresponds to all query plans' first child,
                                              second plan corresponds to all query plans' second child, and so on.
                feat_vec : np.array         : Vectorized query input of shape (subbatch_size x feat_dim).
                total_time : int            : Vectorized prediction targets for each query in the data.
        """
        # The data[0] reference assumes that all query plans have identical structure!
        node_type = data[0]["Node Type"]
        is_subplan = "Subplan Name" in data[0]
        subbatch_size = len(data)

        # Compute feature vector, there are subbatch_size many queries and feat_dim many features.
        # Each node type has its own featurizing function.
        featurize_fn = self.input_func[node_type]
        feat_vec = np.array([featurize_fn(jss) for jss in data])
        # normalize feat_vec
        mean, std = self.mean_range_dict[node_type]
        feat_vec = (feat_vec - mean) / std
        feat_vec = np.array(feat_vec).astype(np.float32)

        # Compute target prediction vector, which is the actual time for each query plan.
        total_time_s = [jss["Actual Total Time"] for jss in data]
        # TODO(WAN): The scale of 100 is inherited from the original QPPNet reimplementation.
        #            Actual Total Time is in seconds, measurable with pg_sleep(seconds).
        #            I do not know why we are dividing by 100,
        #            1000 would make more sense if we wanted this to be milliseconds for example,
        #            but since we aren't changing it anywhere,
        #            I switched it from a class member to a constant here.
        scale = 100
        total_time_arr = np.array(total_time_s).astype(np.float32) / scale

        # Featurize children plans. Note that the data[0] again assumes identical query plan structure!
        child_plan_lst = []
        if "Plans" in data[0]:
            num_children = len(data[0]["Plans"])
            for child_idx in range(num_children):
                child = [jss["Plans"][child_idx] for jss in data]
                child_plan_dict = self.get_input(child)
                child_plan_lst.append(child_plan_dict)

        new_samp_dict = {
            "node_type": node_type,
            "is_subplan": is_subplan,
            "subbatch_size": subbatch_size,
            "children_plan": child_plan_lst,
            "feat_vec": feat_vec,
            "total_time": total_time_arr,
        }
        return new_samp_dict

    ###############################################################################
    #       Sampling subbatch data from the dataset; total size is batch_size     #
    ###############################################################################
    def sample_data(self):
        """
        Randomly sample a batch of data points from the training dataset.

        Returns
        -------
        parsed_input : List[Dict]
            A list of dictionaries, where each dictionary is the vectorized output of the get_input() function,
            where "each dictionary contains all samples in the batch that comes from this group".
            TODO(WAN): What is the old comment saying? It looks like we just sample a bunch of stuff randomly...
        """

        # dataset: all queries used in training

        # Pick a range of indexes to sample.
        samp = self._rng.choice(
            np.arange(self.datasize), self.batch_size, replace=False
        )

        # Construct the sampling groups; for each query, there are num_grps[i] many groups.
        # Then for each query, for each group, we have an empty list to start off with.
        samp_group = [[[] for _ in range(self.num_grps[i])] for i in range(self.num_q)]
        # For each index to be sampled,
        for idx in samp:
            # Get the corresponding group assignments.
            grp_idx = self.grp_idxes[idx]
            # TODO(WAN): I spent ten minutes refactoring out, naming, and staring at this line of code.
            #            This works under the assumption that each contiguous num_sample_per_q block corresponds to
            #            a new unique query, which in practice we have guaranteed by construction above.
            #            I think we can simplify this...
            query_bucket = idx // self.num_sample_per_q
            query_plan = self.dataset[idx]
            samp_group[query_bucket][grp_idx].append(query_plan)

        parsed_input = []
        for i, query_groups in enumerate(samp_group):
            for group in query_groups:
                if len(group) != 0:
                    parsed_input.append(self.get_input(group))

        return parsed_input

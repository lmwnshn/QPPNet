import argparse
import collections
from pprint import pprint

import psycopg


class PgSnapshot:
    """

    Attributes
    ----------
    rel_names : List[str]
        Relation names.

    index_names : List[str]
        Index names.

    rel_attr_list_dict : Dict[str, List[str]]
        Map from a relation name to a list of its attributes.
    """

    def __init__(self, db_name, db_user, db_pass):
        # TODO(WAN): These constants are copied over from the original QPPNet reimplementation.
        #            Specifically, they work for TPC-H, but may need tweaking for completion.
        self.all_dicts = [
            "Aggregate",
            "Gather Merge",
            "Sort",
            "Seq Scan",
            "Index Scan",
            "Index Only Scan",
            "Bitmap Heap Scan",
            "Bitmap Index Scan",
            "Limit",
            "Hash Join",
            "Hash",
            "Nested Loop",
            "Materialize",
            "Merge Join",
            "Subquery Scan",
            "Gather",
        ]
        self.join_types = ["semi", "inner", "anti", "full", "right"]
        self.parent_rel_types = ["inner", "outer", "subquery"]
        self.sort_algos = ["quicksort", "top-n heapsort"]
        self.aggreg_strats = ["plain", "sorted", "hashed"]
        self._snapshot_db(db_name, db_user, db_pass)

    def _snapshot_db(self, db_name, db_user, db_pass):
        """
        Take a snapshot of the database state, populating internal fields.

        Parameters
        ----------
        db_name : str
            Name of database to connect to.

        db_user : str
            Database user to connect as.

        db_pass : str
            Password for the specified database and user.
        """
        with psycopg.connect(
            f"dbname={db_name} user={db_user} password={db_pass}"
        ) as conn:
            with conn.cursor() as cursor:
                rel_names = []
                cursor.execute(
                    "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public';"
                )
                for result in cursor.fetchall():
                    schemaname, tablename = result
                    rel_names.append(tablename)

                index_names = []
                cursor.execute(
                    "SELECT tablename, indexname FROM pg_indexes WHERE tablename NOT LIKE 'pg_%' AND indexname NOT LIKE '%_pkey';"
                )
                for result in cursor.fetchall():
                    tablename, indexname = result
                    index_names.append(indexname)

                rel_attr_list_dict = {}
                for rel in rel_names:
                    attrs = []
                    cursor.execute(
                        f"SELECT attname FROM pg_attribute WHERE attrelid = '{rel}'::regclass::oid "
                        "AND attnum > 0 ORDER BY attnum;"
                    )
                    for result in cursor.fetchall():
                        attname = result[0]
                        attrs.append(attname)
                    rel_attr_list_dict[rel] = attrs

                med_dict, min_dict, max_dict = {}, {}, {}

                def convert(x):
                    # Convert numerical attributes to floats. Otherwise, return 0.
                    try:
                        return float(x)
                    except:
                        return 0

                for rel in rel_attr_list_dict:
                    attrs = rel_attr_list_dict[rel]

                    mins = ", ".join(f"min({attr})" for attr in attrs)
                    cursor.execute(f"SELECT {mins} FROM {rel};")
                    result = cursor.fetchone()
                    min_dict[rel] = [convert(res) for res in result]

                    maxs = ", ".join(f"max({attr})" for attr in attrs)
                    cursor.execute(f"SELECT {maxs} FROM {rel};")
                    result = cursor.fetchone()
                    max_dict[rel] = [convert(res) for res in result]

                    meds = ", ".join(
                        f"percentile_disc(0.5) within group (order by {attr})"
                        for attr in attrs
                    )
                    cursor.execute(f"SELECT {meds} FROM {rel};")
                    result = cursor.fetchone()
                    med_dict[rel] = [convert(res) for res in result]

        attr_val_dict = {"min": min_dict, "max": max_dict, "med": med_dict}
        max_num_attr = max(len(attrs) for attrs in rel_attr_list_dict.values())

        self.num_rel = len(rel_names)
        self.num_index = len(index_names)
        self.rel_names = rel_names
        self.index_names = index_names
        self.rel_attr_list_dict = rel_attr_list_dict
        self.attr_val_dict = attr_val_dict
        self.max_num_attr = max_num_attr

        all_input_funcs = {
            "Hash Join": self.get_join_input,
            "Merge Join": self.get_join_input,
            "Seq Scan": self.get_scan_input,
            "Index Scan": self.get_index_scan_input,
            "Index Only Scan": self.get_index_scan_input,
            "Bitmap Heap Scan": self.get_scan_input,
            "Bitmap Index Scan": self.get_bitmap_index_scan_input,
            "Sort": self.get_sort_input,
            "Hash": self.get_hash_input,
            "Aggregate": self.get_aggreg_input,
        }
        self.all_input_funcs = collections.defaultdict(
            lambda: self.get_basics, all_input_funcs
        )

        len_basics = 3
        len_rel_vec = self.num_rel
        len_rel_attr_vec = self.max_num_attr * 3  # min, median, max
        len_index_vec = self.num_index
        len_sort_key_vec = self.num_rel * self.max_num_attr

        # TODO(WAN): Doublecheck the +32's.
        dim_dict = {
            "Seq Scan": len_basics + len_rel_vec + len_rel_attr_vec,
            "Index Scan": len_basics
            + len_rel_vec
            + len_rel_attr_vec
            + len_index_vec
            + 1,
            "Index Only Scan": len_basics
            + len_rel_vec
            + len_rel_attr_vec
            + len_index_vec
            + 1,
            "Bitmap Heap Scan": len_basics + len_rel_vec + len_rel_attr_vec + 32,
            "Bitmap Index Scan": len_basics + len_index_vec,
            "Sort": len_basics + len_sort_key_vec + len(self.sort_algos) + 32,
            "Hash": len_basics + 1 + 32,
            "Hash Join": len_basics
            + len(self.join_types)
            + len(self.parent_rel_types)
            + 32 * 2,
            "Merge Join": len_basics
            + len(self.join_types)
            + len(self.parent_rel_types)
            + 32 * 2,
            "Aggregate": len_basics + len(self.aggreg_strats) + 1 + 32,
            "Nested Loop": len_basics + 32 * 2,
            "Limit": len_basics + 32,
            "Subquery Scan": len_basics + 32,
            "Materialize": len_basics + 32,
            "Gather Merge": len_basics + 32,
            "Gather": len_basics + 32,
        }
        self.dim_dict = dim_dict

    def get_basics(self, plan_dict):
        # Return plan width, plan rows, total cost.
        # TODO(WAN): there used to be a comment saying that we
        #            "need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket".
        #            Do we still need to do this?
        return [
            plan_dict["Plan Width"],
            plan_dict["Plan Rows"],
            plan_dict["Total Cost"],
        ]

    def get_rel_one_hot(self, rel_name):
        # One-hot encodes the relation.
        arr = [0] * len(self.rel_names)
        arr[self.rel_names.index(rel_name)] = 1
        return arr

    def get_index_one_hot(self, index_name):
        # One-hot encodes the index.
        arr = [0] * len(self.index_names)
        arr[self.index_names.index(index_name)] = 1
        return arr

    def get_rel_attr_one_hot(self, rel_name, filter_line):
        # Get the concatenation of the min, median, and max vectors for the
        # specified relation with the specified attribute.
        # TODO(WAN): Why is filter_line a filter instead of an equality test?
        #            This seems hacky considering that QPPNet cannot support schema change anyway?
        attr_list = self.rel_attr_list_dict[rel_name]

        med_vec, min_vec, max_vec = (
            [0] * self.max_num_attr,
            [0] * self.max_num_attr,
            [0] * self.max_num_attr,
        )

        for idx, attr in enumerate(attr_list):
            if attr in filter_line:
                med_vec[idx] = self.attr_val_dict["med"][rel_name][idx]
                min_vec[idx] = self.attr_val_dict["min"][rel_name][idx]
                max_vec[idx] = self.attr_val_dict["max"][rel_name][idx]
        return min_vec + med_vec + max_vec

    def get_scan_input(self, plan_dict):
        # Get the input to the scan operator.
        # Components:
        #   Basics
        #   One-hot relation
        #   Min/median/max of attribute, in order: Filter, Recheck Cond, else default 0 0 0
        assert plan_dict["Node Type"] in [
            "Seq Scan",
            "Bitmap Heap Scan",
        ], f"Invalid plan dict: {plan_dict}"

        rel_vec = self.get_rel_one_hot(plan_dict["Relation Name"])
        try:
            rel_attr_vec = self.get_rel_attr_one_hot(
                plan_dict["Relation Name"], plan_dict["Filter"]
            )
        except:
            try:
                rel_attr_vec = self.get_rel_attr_one_hot(
                    plan_dict["Relation Name"], plan_dict["Recheck Cond"]
                )
            except:
                if "Filter" in plan_dict:
                    print("************************* default *************************")
                    print(plan_dict)
                rel_attr_vec = [0] * self.max_num_attr * 3

        return self.get_basics(plan_dict) + rel_vec + rel_attr_vec

    def get_index_scan_input(self, plan_dict):
        # Get the input to the index scan operator.
        # Components:
        #   Basics
        #   One-hot relation
        #   Min/median/max of index condition, else default 0 0 0
        #   One-hot index
        #   1 if scan direction is forward, else 0
        assert plan_dict["Node Type"] in [
            "Index Scan",
            "Index Only Scan",
        ], f"Invalid plan dict: {plan_dict}"

        rel_vec = self.get_rel_one_hot(plan_dict["Relation Name"])
        index_vec = self.get_index_one_hot(plan_dict["Index Name"])

        try:
            rel_attr_vec = self.get_rel_attr_one_hot(
                plan_dict["Relation Name"], plan_dict["Index Cond"]
            )
        except:
            if "Index Cond" in plan_dict:
                print(
                    "********************* default rel_attr_vec *********************"
                )
                print(plan_dict)
            rel_attr_vec = [0] * self.max_num_attr * 3

        res = (
            self.get_basics(plan_dict)
            + rel_vec
            + rel_attr_vec
            + index_vec
            + [1 if plan_dict["Scan Direction"] == "Forward" else 0]
        )

        return res

    def get_bitmap_index_scan_input(self, plan_dict):
        # Get the input to the bitmap index scan operator.
        # Components:
        #   Basics
        #   One-hot index
        assert (
            plan_dict["Node Type"] == "Bitmap Index Scan"
        ), f"Invalid plan dict: {plan_dict}"
        index_vec = self.get_index_one_hot(plan_dict["Index Name"])

        return self.get_basics(plan_dict) + index_vec

    def get_hash_input(self, plan_dict):
        # Components:
        #   Basics
        #   Hash buckets
        assert plan_dict["Node Type"] == "Hash", f"Invalid plan dict: {plan_dict}"
        return self.get_basics(plan_dict) + [plan_dict["Hash Buckets"]]

    def get_join_input(self, plan_dict):
        # Components:
        #   One-hot join type.
        #   One-hot parent relation type if applicable, else 0 0 0.
        assert plan_dict["Node Type"] in [
            "Hash Join",
            "Merge Join",
        ], f"Invalid plan dict: {plan_dict}"
        type_vec = [0] * len(self.join_types)
        type_vec[self.join_types.index(plan_dict["Join Type"].lower())] = 1
        par_rel_vec = [0] * len(self.parent_rel_types)
        if "Parent Relationship" in plan_dict:
            par_rel_vec[
                self.parent_rel_types.index(plan_dict["Parent Relationship"].lower())
            ] = 1
        return self.get_basics(plan_dict) + type_vec + par_rel_vec

    def get_sort_key_input(self, plan_dict):
        # Components:
        #   Return a num_rel * max_num_attr long vector,
        #   (basically padding each relation to have max_num_attr),
        #   where all the sort keys in the input plan are set to 1.
        kys = plan_dict["Sort Key"]
        one_hot = [0] * (self.num_rel * self.max_num_attr)
        for key in kys:
            key = key.replace("(", " ").replace(")", " ")
            for subkey in key.split(" "):
                if subkey != " " and "." in subkey:
                    rel_name, attr_name = subkey.split(" ")[0].split(".")
                    if rel_name in self.rel_names:
                        one_hot[
                            self.rel_names.index(rel_name) * self.max_num_attr
                            + self.rel_attr_list_dict[rel_name].index(attr_name.lower())
                        ] = 1
        return one_hot

    def get_sort_input(self, plan_dict):
        # Components:
        #   Basics.
        #   Sort key input.
        #   Sort method.
        assert plan_dict["Node Type"] == "Sort", f"Invalid plan dict: {plan_dict}"
        sort_meth = [0] * len(self.sort_algos)
        if "Sort Method" in plan_dict:
            if "external" not in plan_dict["Sort Method"].lower():
                sort_meth[self.sort_algos.index(plan_dict["Sort Method"].lower())] = 1

        return (
            self.get_basics(plan_dict) + self.get_sort_key_input(plan_dict) + sort_meth
        )

    def get_aggreg_input(self, plan_dict):
        # Components:
        #   Basics.
        #   Aggregation strategy.
        #   1 if parallel aware, else 0.
        assert plan_dict["Node Type"] == "Aggregate", f"Invalid plan dict: {plan_dict}"
        strat_vec = [0] * len(self.aggreg_strats)
        strat_vec[self.aggreg_strats.index(plan_dict["Strategy"].lower())] = 1
        partial_mode_vec = [0] if plan_dict["Parallel Aware"] == "false" else [1]
        return self.get_basics(plan_dict) + strat_vec + partial_mode_vec

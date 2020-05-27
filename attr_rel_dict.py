# basic input:
# plan_width, plan_rows, plan_buffers (ignored), estimated_ios (ignored), total_cost  3

# Sort: sort key [one-hot] (ignored), sort method [one-hot 2];                       2 + 3 = 5
# Hash: Hash buckets, hash algos [one-hot] (ignored);                                1 + 3 = 4
# Hash Join: Join type [one-hot 4], parent relationship [one-hot 3];                 7 + 3 = 10
# Scan: relation name [one-hot ?]; attr min, med, max; [use one-hot instead 16]      8 + 16 + 3 = 27
# Index Scan: never seen one; (Skip)
# Aggregate: Strategy [one-hot 3], partial mode, operator (ignored)                  4 + 3 = 7

# all operators used in tpc-h
all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Limit',
             'Hash Join', 'Hash', 'Nested Loop', 'Materialize', 'Merge Join',
             'Subquery Scan']

join_types = ['semi', 'inner', 'anti', 'full']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-N heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']


rel_names = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp',
             'region', 'supplier']

rel_attr_list_dict = \
{
    'customer':
        ['c_custkey',
         'c_name',
         'c_address',
         'c_nationkey',
         'c_phone',
         'c_acctbal',
         'c_mktsegment',
         'c_comment'],
    'lineitem':
        ['l_orderkey',
         'l_partkey',
         'l_suppkey',
         'l_linenumber',
         'l_quantity',
         'l_extendedprice',
         'l_discount',
         'l_tax',
         'l_returnflag',
         'l_linestatus',
         'l_shipdate',
         'l_commitdate',
         'l_receiptdate',
         'l_shipinstruct',
         'l_shipmode',
         'l_comment'],
    'nation':
        ['n_nationkey',
         'n_name',
         'n_regionkey',
         'n_comment'],
    'orders':
        ['o_orderkey',
         'o_custkey',
         'o_orderstatus',
         'o_totalprice',
         'o_orderdate',
         'o_orderpriority',
         'o_clerk',
         'o_shippriority',
         'o_comment'],
    'part':
        ['p_partkey',
         'p_name',
         'p_mfgr',
         'p_brand',
         'p_type',
         'p_size',
         'p_container',
         'p_retailprice',
         'p_comment'],
    'partsupp':
        ['ps_partkey',
         'ps_suppkey',
         'ps_availqty',
         'ps_supplycost',
         'ps_comment'],
    'region':
        ['r_regionkey',
         'r_name',
         'r_comment'],
    'supplier':
        ['s_suppkey',
         's_name',
         's_address',
         's_nationkey',
         's_phone',
         's_acctbal',
         's_comment']
}
import os
from uuid import uuid4

import pandas as pd
import pytest

import splink.postgres.postgres_comparison_library as cl
from splink.postgres.postgres_linker import PostgresLinker

from .basic_settings import get_settings_dict
from .linker_utils import _test_table_registration, register_roc_data


def test_full_example_postgres(tmp_path, pg_conn):
    df = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    settings_dict = get_settings_dict()

    # Avoid fuzzy matching for time being
    settings_dict["comparisons"][0] = cl.exact_match("first_name")

    linker = PostgresLinker(
        df,
        connection=pg_conn,
    )
    linker.load_settings(settings_dict)

    linker.count_num_comparisons_from_blocking_rule(
        'l.first_name = r.first_name and l."surname" = r."surname"'
    )
    linker.cumulative_num_comparisons_from_blocking_rules_chart(
        [
            'l.first_name = r.first_name',
            'l.surname = r.surname',
            'l.city = r.city',
        ]
    )

    # TODO: fix bug and restore:
    # linker.profile_columns(
    #     [
    #         "first_name",
    #         '"surname"',
    #         'first_name || "surname"',
    #         "concat(city, first_name)",
    #     ]
    # )
    linker.missingness_chart()
    linker.compute_tf_table("city")
    linker.compute_tf_table("first_name")

    linker.estimate_u_using_random_sampling(max_pairs=1e6, seed=1)
    linker.estimate_probability_two_random_records_match(
        ["l.email = r.email"], recall=0.3
    )
    # try missingness chart again now that concat_with_tf is precomputed
    linker.missingness_chart()

    blocking_rule = 'l.first_name = r.first_name and l."surname" = r."surname"'
    linker.estimate_parameters_using_expectation_maximisation(blocking_rule)

    blocking_rule = "l.dob = r.dob"
    linker.estimate_parameters_using_expectation_maximisation(blocking_rule)

    df_predict = linker.predict()

    linker.comparison_viewer_dashboard(
        df_predict, os.path.join(tmp_path, "test_scv_postgres.html"), True, 2
    )

    df_e = df_predict.as_pandas_dataframe(limit=5)
    records = df_e.to_dict(orient="records")
    linker.waterfall_chart(records)

    register_roc_data(linker)
    linker.roc_chart_from_labels_table("labels")

    df_clusters = linker.cluster_pairwise_predictions_at_threshold(df_predict, 0.1)

    linker.cluster_studio_dashboard(
        df_predict,
        df_clusters,
        sampling_method="by_cluster_size",
        out_path=os.path.join(tmp_path, "test_cluster_studio.html"),
    )

    # TODO: fix bug and restore:
    # linker.unlinkables_chart(source_dataset="Testing")

    # TODO: fix bug and restore:
    # _test_table_registration(linker)

    record = {
        "unique_id": 1,
        "first_name": "John",
        "surname": "Smith",
        "dob": "1971-05-24",
        "city": "London",
        "email": "john@smith.net",
        "group": 10000,
    }

    # TODO: fix bug and restore:
    # linker.find_matches_to_new_records(
    #     [record], blocking_rules=[], match_weight_threshold=-10000
    # )

    # Test saving and loading
    path = os.path.join(tmp_path, "model.json")
    linker.save_settings_to_json(path)

    linker_2 = PostgresLinker(df, connection=pg_conn)
    linker_2.load_settings(path)
    linker_2.load_settings_from_json(path)


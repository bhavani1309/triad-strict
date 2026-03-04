import os
import numpy as np # type: ignore

from ir_models.vsm import VSMModel
from evaluation.metrics import mean_average_precision
from triad_core.transitive_links import TransitiveEngine
from triad_core.raw_enrichment import RawEnrichmentEngine
from preprocessing.raw_biterm_loader import (
    load_requirements_biterm,
    load_code_biterm
)


REQ_PATH = "dataset/dronology/processed/req"
DD_PATH = "dataset/dronology/processed/design_definition"
CODE_PATH = "dataset/dronology/processed/code"
GROUND_TRUTH_PATH = "dataset/dronology/trace_matrices/req-code.txt"


def load_ground_truth(path):
    gt = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gt.add((parts[0], parts[1]))
    return gt


def run_strict_biterm_triad_vsm():

    print("Loading biterm artifacts (generated from raw)...")

    req = load_requirements_biterm(REQ_PATH)
    dd = load_requirements_biterm(DD_PATH)
    code = load_code_biterm(CODE_PATH)

    print(f"Req: {len(req)}  DD: {len(dd)}  Code: {len(code)}")

    # Req–DD
    print("\nBuilding Req–DD similarity...")
    vsm_si = VSMModel()
    vsm_si.build_dual_corpus(req, dd)
    sim_si = vsm_si.similarity_dual()

    # Enrichment
    print("Enriching requirements...")
    enrich_engine = RawEnrichmentEngine(t=3)
    req_enriched = enrich_engine.enrich_requirements(req, dd, sim_si)

    # Req–Code
    print("Building enriched Req–Code similarity...")
    vsm_st = VSMModel()
    vsm_st.build_dual_corpus(req_enriched, code)
    sim_st = vsm_st.similarity_dual()

    # DD–Code
    print("Building DD–Code similarity...")
    vsm_it = VSMModel()
    vsm_it.build_dual_corpus(dd, code)
    sim_it = vsm_it.similarity_dual()

    # Req–Req
    print("Building Req–Req similarity...")
    vsm_ss = VSMModel()
    vsm_ss.build_single_corpus(req_enriched)
    sim_ss = vsm_ss.similarity_single()

    # DD–DD
    print("Building DD–DD similarity...")
    vsm_ii = VSMModel()
    vsm_ii.build_single_corpus(dd)
    sim_ii = vsm_ii.similarity_single()

    # Transitive propagation
    print("Applying transitive propagation...")
    trans_engine = TransitiveEngine(base_t=3, base_m=0.5)

    outer_bonus = trans_engine.compute_outer_transitive_bonus(
        s_i_matrix=sim_si,
        i_t_matrix=sim_it
    )

    inner_bonus = trans_engine.compute_inner_transitive_bonus(
        s_s_matrix=sim_ss,
        s_i_matrix=sim_si,
        i_i_matrix=sim_ii,
        i_t_matrix=sim_it
    )

    bonus_matrix = outer_bonus + inner_bonus

    # Paper-style multiplicative score adjustment
    adjusted_sim = sim_st * (1 + bonus_matrix)

    print("\nEvaluating MAP...")
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    source_ids = list(req.keys())
    target_ids = list(code.keys())

    map_score = mean_average_precision(
        adjusted_sim,
        source_ids,
        target_ids,
        ground_truth
    )

    print("\n===== STRICT BITERM TRIAD (VSM) RESULT =====")
    print("MAP:", map_score)


if __name__ == "__main__":
    run_strict_biterm_triad_vsm()
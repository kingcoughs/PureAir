"""
Counterfactual Policy Simulator (do-Calculus Engine)
Evaluates what-if regulatory interventions (Odd-Even, Truck bans, Construction halts, Smog guns)
by constraining input feature vectors and running forward simulation through Model 1.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from app.config import grid_config
from app.grid.h3_grid import grid_manager
from app.data.dataset_builder import dataset_builder
from app.models.st_gnn import model1_lsp

class PolicySimulator:
    """
    Evaluates policy interventions using do-calculus on the Spatio-Temporal Graph:
    Y_policy = Model1(X with do(X_target = kappa * X_target), A(t))
    """
    def __init__(self, model1=None):
        self.model1 = model1 or model1_lsp

    def simulate_interventions(
        self,
        odd_even_active: bool = False,
        truck_diversion_active: bool = False,
        construction_halt_active: bool = False,
        industrial_curfew_active: bool = False,
        smog_guns_units: int = 0
    ) -> Dict[str, Any]:
        """
        Runs before-and-after counterfactual simulation.
        
        Args:
            odd_even_active: 50% private vehicle traffic reduction
            truck_diversion_active: Non-destined heavy diesel commercial trucks diverted to EPE/WPE (35% drop)
            construction_halt_active: 90% halt on dust-emitting building & demolition works
            industrial_curfew_active: 60% reduction in industrial boiler operations
            smog_guns_units: Number of deployed anti-smog mist cannons (0 - 200 units)
        """
        # 1. Base run (no intervention)
        X_base, A_base = dataset_builder.build_current_node_features()
        X_seq_base = np.repeat(X_base[np.newaxis, :, :], 12, axis=0)
        
        base_preds, _, _ = self.model1.forward(X_seq_base, A_base)
        base_aqi_now = base_preds[:, 0] # [N] (1h horizon)
        base_aqi_6h = base_preds[:, 2]  # [N] (6h horizon)

        # 2. Calculate policy factor multipliers (do-calculus constraints)
        traffic_factor = 1.0
        if odd_even_active:
            traffic_factor *= 0.50
        if truck_diversion_active:
            traffic_factor *= 0.65

        construction_factor = 0.10 if construction_halt_active else 1.0
        industrial_factor = 0.40 if industrial_curfew_active else 1.0
        
        # Smog gun dust knockdown factor: 1 unit ~ 0.1% localized dust suppression, up to 25% max
        dust_factor = max(0.75, 1.0 - (smog_guns_units * 0.0012))

        # 3. Counterfactual run with do(X_target = kappa * X_target)
        X_policy, A_policy = dataset_builder.build_current_node_features(
            custom_traffic_factor=traffic_factor,
            custom_industrial_factor=industrial_factor,
            custom_construction_factor=construction_factor,
            custom_dust_factor=dust_factor
        )
        X_seq_policy = np.repeat(X_policy[np.newaxis, :, :], 12, axis=0)

        policy_preds, _, _ = self.model1.forward(X_seq_policy, A_policy)
        policy_aqi_now = policy_preds[:, 0]
        policy_aqi_6h = policy_preds[:, 2]

        # 4. Compute delta AQI reductions per node and identify largest beneficiaries
        node_impacts = []
        for i, hex_id in enumerate(grid_manager.hex_ids):
            node = grid_manager.nodes[hex_id]
            b_val = float(base_aqi_6h[i])
            p_val = float(policy_aqi_6h[i])
            delta = b_val - p_val
            pct_drop = (delta / (b_val + 1e-6)) * 100.0

            node_impacts.append({
                "hex_id": hex_id,
                "name": node.name,
                "zone": node.zone,
                "baseline_aqi_6h": int(round(b_val)),
                "projected_aqi_6h": int(round(p_val)),
                "delta_aqi_drop": int(round(delta)),
                "percentage_reduction": round(pct_drop, 1),
                "estimated_lag_hours": self._estimate_time_lag(node, odd_even_active, construction_halt_active)
            })

        # Sort by largest point reduction
        node_impacts.sort(key=lambda x: x["delta_aqi_drop"], reverse=True)

        # Citywide aggregate impact
        avg_base_6h = float(np.mean(base_aqi_6h))
        avg_policy_6h = float(np.mean(policy_aqi_6h))
        avg_delta = avg_base_6h - avg_policy_6h

        active_interventions = []
        if odd_even_active: active_interventions.append("Odd-Even Traffic Rationing (-50% traffic)")
        if truck_diversion_active: active_interventions.append("Heavy Commercial Freight Bypass to EPE/WPE")
        if construction_halt_active: active_interventions.append("Halt Tier-1 & Tier-2 Construction Works")
        if industrial_curfew_active: active_interventions.append("50% Industrial Capacity & Boilers Curfew")
        if smog_guns_units > 0: active_interventions.append(f"Deploy {smog_guns_units} Anti-Smog Water Mist Units")

        return {
            "simulation_timestamp": time.time(),
            "active_interventions": active_interventions or ["No Active Interventions (Baseline)"],
            "citywide_summary": {
                "baseline_mean_aqi_6h": int(round(avg_base_6h)),
                "projected_mean_aqi_6h": int(round(avg_policy_6h)),
                "average_delta_reduction": int(round(avg_delta)),
                "average_percentage_drop": round((avg_delta / (avg_base_6h + 1e-6)) * 100.0, 1)
            },
            "top_beneficiary_wards": node_impacts[:6],
            "all_ward_impacts": node_impacts
        }

    def _estimate_time_lag(self, node: Any, is_traffic: bool, is_construction: bool) -> int:
        """Estimates physical lag (in hours) before atmospheric air reflects the policy change."""
        if is_traffic and node.traffic_weight > 0.6:
            return 3 # Traffic impacts appear in 2-3 hours
        elif is_construction and node.construction_weight > 0.5:
            return 6 # Construction settling takes 5-7 hours
        else:
            return 4

policy_simulator = PolicySimulator()


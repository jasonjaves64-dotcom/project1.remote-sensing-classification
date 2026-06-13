"""V6 Mathematical Theory — Database utilities for geometric/topological metrics.

Extends db_utils_edl.py with methods for:
  - Geometric invariants tracking (K, H, kappa_1, kappa_2, tau_g)
  - Topological conflict metrics (Noise/Structural/HighOrder counts)
  - TTA safety logs (intervention level, gradient alignment, recovery)
  - Cross-module synergy metrics
"""
import os
import json
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional


class V6MathMetricsDB:
    """Database access layer for V6 mathematical theory metrics."""

    def __init__(self, host='localhost', database='crop_classification',
                 user=None, password=None, port=3306):
        self.host = host
        self.database = database
        self.user = user or os.environ.get('MYSQL_USER', 'root')
        self.password = password if password is not None else os.environ.get('MYSQL_PASSWORD', '')
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host, database=self.database,
                user=self.user, password=self.password, port=self.port
            )
            return self.connection.is_connected()
        except Error as e:
            print(f"V6MathMetricsDB connect failed: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def _execute(self, query: str, params: tuple = None) -> int:
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params) if params else cursor.execute(query)
            self.connection.commit()
            return cursor.lastrowid or cursor.rowcount
        except Error as e:
            print(f"Query failed: {e}")
            self.connection.rollback()
            return -1
        finally:
            if cursor:
                cursor.close()

    def _fetch(self, query: str, params: tuple = None) -> List[Dict]:
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params) if params else cursor.execute(query)
            return cursor.fetchall()
        except Error as e:
            print(f"Fetch failed: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    # ── Geometric Invariants ──

    def add_geometric_invariants(self, exp_id: int, sample_id: str,
                                  invariants: dict) -> int:
        """Insert geometric invariant metrics for a sample."""
        query = """
        INSERT INTO geometric_invariants
        (exp_id, sample_id, K_mean, K_std, H_mean, H_std,
         k1_mean, k1_std, k2_mean, k2_std, tau_g_mean, tau_g_std,
         K_se3_deviation, elev_min, elev_max, elev_range, slope_mean, slope_max)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            exp_id, sample_id,
            invariants.get('K_mean'), invariants.get('K_std'),
            invariants.get('H_mean'), invariants.get('H_std'),
            invariants.get('k1_mean'), invariants.get('k1_std'),
            invariants.get('k2_mean'), invariants.get('k2_std'),
            invariants.get('tau_g_mean'), invariants.get('tau_g_std'),
            invariants.get('K_se3_deviation'),
            invariants.get('elev_min'), invariants.get('elev_max'),
            invariants.get('elev_range'),
            invariants.get('slope_mean'), invariants.get('slope_max'),
        )
        return self._execute(query, params)

    # ── Topological Conflict ──

    def add_conflict_metrics(self, exp_id: int, epoch: int, phase: int,
                              metrics: dict) -> int:
        """Insert topological conflict classification metrics."""
        query = """
        INSERT INTO topological_conflict_metrics
        (exp_id, epoch, phase, noise_count, structural_count, high_order_count,
         avg_kappa, avg_h1_norm, conflict_ratio, avg_persistence,
         n_long_lived_features, per_class_conflict)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        per_class_json = json.dumps(metrics.get('per_class_conflict', {}))
        params = (
            exp_id, epoch, phase,
            metrics.get('noise_count', 0),
            metrics.get('structural_count', 0),
            metrics.get('high_order_count', 0),
            metrics.get('avg_kappa'),
            metrics.get('avg_h1_norm'),
            metrics.get('conflict_ratio'),
            metrics.get('avg_persistence'),
            metrics.get('n_long_lived_features'),
            per_class_json,
        )
        return self._execute(query, params)

    # ── TTA Safety ──

    def add_tta_safety_log(self, exp_id: int, step: int,
                            monitors: dict, action: dict) -> int:
        """Log a TTA safety monitoring event."""
        query = """
        INSERT INTO tta_safety_logs
        (exp_id, step, gradient_alignment, semantic_map, cohomology_conflict,
         intervention_level, action_taken, lr_factor, auto_recovered, recovery_steps)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            exp_id, step,
            monitors.get('gradient_alignment'),
            monitors.get('semantic_map'),
            monitors.get('cohomology_conflict'),
            monitors.get('intervention_level', 0),
            action.get('action', 'normal'),
            action.get('new_lr_factor', 1.0),
            1 if action.get('intervention_level', 0) == 0 and monitors.get('prev_level', 0) > 0 else 0,
            monitors.get('recovery_steps'),
        )
        return self._execute(query, params)

    # ── Cross-Module Synergy ──

    def add_synergy_metrics(self, exp_id: int, metrics: dict) -> int:
        """Insert cross-module synergy metrics."""
        query = """
        INSERT INTO cross_module_synergy
        (exp_id, geo_anchor_alignment_error, geo_conflict_detection_precision,
         T_eff_estimated, T_max_safe_steps, topo_parameter_protection_ratio,
         adaptive_regularization_alpha)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            exp_id,
            metrics.get('geo_anchor_alignment_error'),
            metrics.get('geo_conflict_detection_precision'),
            metrics.get('T_eff_estimated'),
            metrics.get('T_max_safe_steps'),
            metrics.get('topo_parameter_protection_ratio'),
            metrics.get('adaptive_regularization_alpha'),
        )
        return self._execute(query, params)

    # ── Query Methods ──

    def get_geometric_invariants(self, exp_id: int) -> List[Dict]:
        return self._fetch(
            "SELECT * FROM geometric_invariants WHERE exp_id = %s ORDER BY created_at DESC",
            (exp_id,))

    def get_conflict_metrics(self, exp_id: int) -> List[Dict]:
        return self._fetch(
            "SELECT * FROM topological_conflict_metrics WHERE exp_id = %s ORDER BY epoch",
            (exp_id,))

    def get_tta_safety_logs(self, exp_id: int = None, limit: int = 100) -> List[Dict]:
        if exp_id:
            return self._fetch(
                "SELECT * FROM tta_safety_logs WHERE exp_id = %s ORDER BY step DESC LIMIT %s",
                (exp_id, limit))
        return self._fetch(
            "SELECT * FROM tta_safety_logs ORDER BY created_at DESC LIMIT %s",
            (limit,))

    def get_synergy_metrics(self, exp_id: int) -> List[Dict]:
        return self._fetch(
            "SELECT * FROM cross_module_synergy WHERE exp_id = %s ORDER BY created_at DESC",
            (exp_id,))

    def get_experiment_v6_config(self, exp_id: int) -> Dict:
        rows = self._fetch(
            "SELECT model_version, siren_dem_enabled, geometric_invariants_enabled, "
            "topological_evidence_enabled, grassmann_alignment_enabled, tta_enabled, "
            "spectral_balanced_init FROM experiments WHERE exp_id = %s",
            (exp_id,))
        return rows[0] if rows else {}

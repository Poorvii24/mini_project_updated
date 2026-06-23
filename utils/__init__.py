from .preprocessing import preprocess, build_features, load_scaler, FEATURE_COLS, _safe_port
from .shap_explainer import (
    compute_shap_values,
    explain_single_alert,
    explain_batch,
    global_importance,
    FEATURE_DESCRIPTIONS,
)
from .mitre_mapper import (
    map_flow_to_techniques,
    map_dataframe,
    tactic_summary,
    technique_summary,
    TECHNIQUES,
    TACTIC_COLORS,
)
from .auth import (
    load_auth_config,
    get_authenticator,
    get_current_role,
    has_permission,
    require_permission,
    render_login,
    render_user_sidebar,
)
from .virustotal import (
    lookup_ip,
    lookup_ips_batch,
    get_verdict,
    render_vt_results,
)
from .abuseipdb import (
    lookup_ip       as abuseipdb_lookup_ip,
    lookup_ips_batch as abuseipdb_lookup_batch,
    get_verdict     as abuseipdb_get_verdict,
    render_abuseipdb_results,
)
from .packet_capture import (
    FlowBuffer,
    get_flow_buffer,
    list_network_interfaces,
    start_live_capture,
    start_csv_replay,
    stop_capture,
    SCAPY_AVAILABLE,
)
from .live_pipeline import (
    LiveDetectionPipeline,
    get_pipeline,
)

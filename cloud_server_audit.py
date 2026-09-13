from cloud_server import app
from audit_extension import install as install_audit
from v4_extension import install as install_v4

install_audit(app)
install_v4(app)

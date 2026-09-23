# CMN-C1-051 — /health readiness check
#
# Finding (2026-08-19, Medium): a provision_secrets() failure at startup was
# only logged -- the process still finished booting and /health always
# returned {"status": "ok"}, so a deployment readiness probe could not
# distinguish "booted with a working secrets provider" from "booted with a
# broken one." /health now returns 503 + status="degraded" when secrets
# provisioning failed at startup.


class TestServerHealth:
    def test_health_returns_agent_identity(self):
        import src.api.server as server

        response = server.health()
        assert response == {"status": "ok", "agent": "CMN-C1-051"}

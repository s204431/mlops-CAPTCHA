from locust import HttpUser, between, task


class User(HttpUser):
    """Locust user"""

    wait_time = between(1, 2)

    @task()
    def run_on_frontend(self) -> None:
        """Get the frontend."""
        self.client.get("/")

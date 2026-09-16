import os

import hypothesis

hypothesis.settings.register_profile("dev", max_examples=100)
hypothesis.settings.register_profile("ci", max_examples=500, deadline=None)
hypothesis.settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
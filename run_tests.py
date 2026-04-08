"""Quick test runner that prints results clearly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

from test_environment import *

classes = [
    TestEnvironmentReset,
    TestEnvironmentStep,
    TestEnvironmentState,
    TestStatusClassificationGrader,
    TestPriorityAssignmentGrader,
    TestDecisionMakingGrader,
    TestRewardFunction,
]

passed = 0
failed = 0

for cls in classes:
    obj = cls()
    methods = [m for m in dir(obj) if m.startswith("test_")]
    for m in methods:
        try:
            getattr(obj, m)()
            passed += 1
            print(f"  PASS  {cls.__name__}.{m}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {cls.__name__}.{m}: {e}")

print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(1 if failed else 0)

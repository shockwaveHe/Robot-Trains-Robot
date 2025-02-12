import argparse
import importlib

import jax.numpy as jnp

# This script is used to test the installation of torch and jax, and their ability to run on CUDA.


def verify_package(pkg_name, test_fn):
    """Function to verify if a package is installed and can perform a basic operation."""
    try:
        module = importlib.import_module(pkg_name)
        print(f"✅ {pkg_name} is installed and imported successfully.")
        test_fn(module)
    except ImportError as e:
        print(f"❌ {pkg_name} is missing or failed to import: {str(e)}")
    except Exception as e:
        print(f"❌ {pkg_name} failed during operation: {str(e)}")


def test_torch(torch, run_cuda):
    """Test if torch can run on CUDA."""
    try:
        if run_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            x = torch.rand(3, 3).to(device)
            print(f"✅ Torch CUDA test passed: Tensor created on CUDA:\n{x}")
        else:
            if run_cuda:
                print("❌ Torch CUDA test failed: CUDA is not available.")
            else:
                print("Skipping CUDA test for Torch.")
    except Exception as e:
        print(f"❌ Torch test failed: {e}")


def test_jax(jax, run_cuda):
    """Test if JAX can run on CUDA. Skip CUDA test for Jetson."""
    try:
        if run_cuda and jax.devices()[0].platform == "gpu":
            x = jax.device_put(jnp.array([1, 2, 3]))
            print(f"✅ JAX CUDA test passed: Array created on GPU:\n{x}")
        else:
            if run_cuda:
                print("❌ JAX CUDA test failed: Not using a GPU device.")
            else:
                print("Skipping CUDA test for JAX.")
    except Exception as e:
        print(f"❌ JAX test failed: {e}")


def main(platform_name):
    print(f"Running tests for platform: {platform_name}")

    # Platform-specific behavior
    if platform_name == "linux":
        verify_package("torch", lambda torch: test_torch(torch, True))
        verify_package("jax", lambda jax: test_jax(jax, True))
    elif platform_name == "jetson":
        verify_package("torch", lambda torch: test_torch(torch, True))
        verify_package("jax", lambda jax: test_jax(jax, False))
    elif platform_name in ["macos", "windows", "steam_deck", "rog_ally_x"]:
        print(f"No tests for {platform_name}.")
    else:
        print(f"❌ Unknown platform: {platform_name}")


if __name__ == "__main__":
    # Parse platform input
    parser = argparse.ArgumentParser(
        description="Verify Torch and JAX installations and CUDA availability."
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["linux", "macos", "windows", "jetson", "steam_deck", "rog_ally_x"],
        help="Specify the platform.",
    )

    args = parser.parse_args()

    main(args.platform)

use std::path::{Path, PathBuf};
use std::process::Command;

fn find_cuda_root() -> PathBuf {
    if let Ok(path) = std::env::var("CUDA_HOME") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return PathBuf::from(path);
    }
    if Path::new("/usr/local/cuda-12.6").exists() {
        return PathBuf::from("/usr/local/cuda-12.6");
    }
    if Path::new("/usr/local/cuda").exists() {
        return PathBuf::from("/usr/local/cuda");
    }
    PathBuf::from("/usr/local/cuda")
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let kernels_dir = repo_root.join("kernels");
    let kernels_build_dir = kernels_dir.join("build");

    let cuda_root = find_cuda_root();
    let cuda_lib64 = cuda_root.join("lib64");

    // Build CUDA static library using CMake
    std::fs::create_dir_all(&kernels_build_dir).expect("Failed to create kernels/build");

    let cmake_status = Command::new("cmake")
        .current_dir(&kernels_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCUDA_TOOLKIT_ROOT_DIR={}", cuda_root.display()))
        .arg("..")
        .status()
        .expect("Failed to execute cmake for CUDA kernels");

    if !cmake_status.success() {
        panic!("CMake configuration failed for kernels");
    }

    let build_status = Command::new("cmake")
        .current_dir(&kernels_build_dir)
        .args(["--build", ".", "-j"])
        .status()
        .expect("Failed to build CUDA kernels");

    if !build_status.success() {
        panic!("CUDA kernel compilation failed");
    }

    // Instruct Cargo to link the generated static library and CUDA runtime
    println!("cargo:rustc-link-search=native={}", kernels_build_dir.display());
    println!("cargo:rustc-link-lib=static=gpukernels");

    if cuda_lib64.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Re-run triggers
    println!("cargo:rerun-if-changed={}", kernels_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", kernels_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", kernels_dir.join("src").display());
}

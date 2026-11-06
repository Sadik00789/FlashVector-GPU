use std::path::{Path, PathBuf};
use std::process::Command;

fn cuda_lib_dir(root: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        root.join("lib").join("x64")
    } else {
        root.join("lib64")
    }
}

fn is_cuda_toolkit(root: &Path) -> bool {
    let (nvcc, cudart) = if cfg!(target_os = "windows") {
        ("nvcc.exe", "cudart.lib")
    } else {
        ("nvcc", "libcudart.so")
    };
    root.join("bin").join(nvcc).is_file() && cuda_lib_dir(root).join(cudart).is_file()
}

fn nvcc_root_on_path() -> Option<PathBuf> {
    let nvcc = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .find(|dir| dir.join(nvcc).is_file())
            .and_then(|dir| dir.parent().map(Path::to_path_buf))
    })
}

fn find_cuda_root() -> PathBuf {
    let env_candidates = ["CUDA_HOME", "CUDA_PATH"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .map(PathBuf::from);
    let path_candidate = nvcc_root_on_path();
    let default_candidates = [
        PathBuf::from("/usr/local/cuda-12.6"),
        PathBuf::from("/usr/local/cuda"),
    ];

    for candidate in env_candidates
        .chain(path_candidate)
        .chain(default_candidates)
    {
        if is_cuda_toolkit(&candidate) {
            return candidate;
        }
    }

    if cfg!(target_os = "windows") {
        let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut candidates: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| is_cuda_toolkit(path))
                .collect();
            candidates.sort();
            if let Some(path) = candidates.pop() {
                return path;
            }
        }
    }

    panic!(
        "CUDA toolkit not found: expected nvcc and cudart in a toolkit root. \
         Set CUDA_HOME to a CUDA toolkit directory (for example the pip `nvidia/cu13` root)."
    );
}

fn command_available(name: &str, version_flag: &str) -> bool {
    Command::new(name)
        .arg(version_flag)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let kernels_dir = repo_root.join("kernels");
    let kernels_build_dir = kernels_dir.join("build");

    let cuda_root = find_cuda_root();
    let windows = cfg!(target_os = "windows");
    let cuda_lib = cuda_lib_dir(&cuda_root);

    // Build CUDA static library using CMake
    std::fs::create_dir_all(&kernels_build_dir).expect("Failed to create kernels/build");

    let cuda_root_arg = cuda_root.display().to_string().replace('\\', "/");
    let mut configure = Command::new("cmake");
    configure
        .current_dir(&kernels_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCUDA_TOOLKIT_ROOT_DIR={}", cuda_root_arg));

    if windows
        && std::env::var_os("CMAKE_GENERATOR").is_none()
        && command_available("ninja", "--version")
    {
        configure.args(["-G", "Ninja"]);
    }

    let cmake_status = configure
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
    println!(
        "cargo:rustc-link-search=native={}",
        kernels_build_dir.display()
    );
    println!("cargo:rustc-link-lib=static=gpukernels");

    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !windows {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Re-run triggers
    println!(
        "cargo:rerun-if-changed={}",
        kernels_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernels_dir.join("include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernels_dir.join("src").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}

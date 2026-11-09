//! Driver-free compile checks for the cuTile kernels (feature `cuda`).
//!
//! Each kernel is lowered through cuTile's front end to Tile IR and
//! serialized to bytecode for every GPU architecture we target. This needs
//! CUDA headers at build time but **no GPU and no driver**, so it runs in CI
//! and catches kernel type/shape errors that only surface at JIT time.
//!
//! If NVIDIA's `tileiras` assembler is available (it ships in the CUDA
//! toolkit's `bin/`, and in the `nvidia-cuda-tileiras` PyPI wheel), the
//! bytecode is also assembled to a cubin for each target: real GPU machine
//! code, still without a GPU. Set `S2S_REQUIRE_TILEIRAS=1` to make a missing
//! `tileiras` an error instead of a skip (CI does).

#![cfg(feature = "cuda")]

use cutile::compile_api::KernelCompiler;
use cutile_compiler::cuda_tile_runtime_utils::{compile_tile_ir_module, tileiras_binary};
use cutile_ir::bytecode::BytecodeVersion;
use s2s_kernels::cuda::kernels;

/// Architectures from the plan (§7.1): Ampere, Ada, Hopper, Blackwell.
const TARGETS: &[&str] = &["sm_80", "sm_89", "sm_90", "sm_100"];

/// Tile IR version to emit. Pinning it means bytecode serialization does not
/// need to ask a local `tileiras` binary which version it accepts. 13.3 is the
/// minimum that covers every target above (Hopper requires 13.3; cuTile's
/// README lists the per-architecture minimums).
const TILE_IR: BytecodeVersion = BytecodeVersion::V13_3;

/// cuTile's compiler recurses deeply; run on a big stack like its own tests.
fn with_big_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("compile thread panicked");
}

/// Whether to assemble cubins. Panics if `S2S_REQUIRE_TILEIRAS=1` and
/// `tileiras` cannot be found.
fn assemble_cubins() -> bool {
    let found = std::process::Command::new(tileiras_binary())
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !found && std::env::var("S2S_REQUIRE_TILEIRAS").as_deref() == Ok("1") {
        panic!(
            "S2S_REQUIRE_TILEIRAS=1 but tileiras was not found at {:?}",
            tileiras_binary()
        );
    }
    if !found {
        eprintln!("tileiras not found; skipping cubin assembly");
    }
    found
}

fn compile(function: &str, generics: &[usize], strides: &[(&str, &[i32])]) {
    let strides: Vec<(String, Vec<i32>)> = strides
        .iter()
        .map(|(n, s)| (n.to_string(), s.to_vec()))
        .collect();
    let function = function.to_string();
    let generics: Vec<String> = generics.iter().map(|g| g.to_string()).collect();
    let assemble = assemble_cubins();
    with_big_stack(move || {
        let strides: Vec<(&str, &[i32])> = strides
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        for target in TARGETS {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", &function)
                .generics(generics.clone())
                .strides(&strides)
                .target(target)
                .bytecode_version(TILE_IR)
                .compile()
                .unwrap_or_else(|e| panic!("{function} failed to compile for {target}: {e:?}"));
            assert!(
                artifacts.ir_text().contains(&function),
                "IR for {function} does not mention the kernel"
            );
            let bytecode = artifacts
                .bytecode()
                .unwrap_or_else(|e| panic!("{function} bytecode for {target}: {e:?}"));
            assert!(!bytecode.is_empty());
            if assemble {
                // The same path cuTile's JIT takes at launch: serialize at the
                // version `tileiras` reports, link default tile libraries,
                // assemble.
                let cubin =
                    compile_tile_ir_module(artifacts.module(), target).unwrap_or_else(|e| {
                        panic!("tileiras failed for {function} on {target}: {e:?}")
                    });
                assert_eq!(
                    &cubin[..4],
                    b"\x7fELF",
                    "{function}/{target}: cubin is not an ELF file"
                );
            }
        }
    });
}

#[test]
fn vec_add_compiles() {
    compile(
        "vec_add_f32",
        &[128],
        &[("z", &[1]), ("x", &[1]), ("y", &[1])],
    );
}

#[test]
fn rms_norm_fwd_compiles() {
    // N = 1024, BLOCK = 256, row-major [rows, N] inputs.
    compile(
        "rms_norm_fwd_f32",
        &[1024, 256],
        &[
            ("x", &[1024, 1]),
            ("w", &[1]),
            ("y", &[1024, 1]),
            ("rstd", &[1]),
        ],
    );
}

#[test]
fn swiglu_fwd_compiles() {
    compile(
        "swiglu_fwd_f32",
        &[256],
        &[("h", &[1]), ("gate", &[1]), ("up", &[1])],
    );
}

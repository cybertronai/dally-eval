{
  description = "dally-eval: Bill Dally IR evaluator dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            toolchain
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            (pkgs.python3.withPackages (ps: [ ps.numpy ]))
          ];
          RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
          # headless GPU access: RADV ICD + loader + libudev
          VK_ICD_FILENAMES = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
          VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";
          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib:${pkgs.systemd}/lib";
        };
      });
}

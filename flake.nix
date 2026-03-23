{
  description = "RFCellClassificator v2.0 — dev shell bioinformatica";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = false;
        };

        # Python con tutte le dipendenze bioinformatica + ML
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # ── Core ML ────────────────────────────────────────────────
          scikit-learn
          numpy
          scipy
          pandas

          # ── scRNA-seq ───────────────────────────────────────────────
          anndata
          # scanpy non è sempre in nixpkgs unstable, fallback a pip
          # scanpy

          # ── TUI ─────────────────────────────────────────────────────
          rich

          # ── Sistema ─────────────────────────────────────────────────
          psutil

          # ── Sviluppo ────────────────────────────────────────────────
          ipython
          pytest
          black
          mypy
        ]);

      in {
        # ── Dev shell: `nix develop` ────────────────────────────────────
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv

            # BLAS/LAPACK ottimizzati — accelera numpy/scipy su Pi 5
            pkgs.openblas

            # HDF5 per AnnData .h5ad
            pkgs.hdf5

            # Rust toolchain per il futuro backend Lloyd (punto 3)
            pkgs.rustup
            pkgs.pkg-config

            # Util
            pkgs.git
            pkgs.htop
          ];

          shellHook = ''
            echo ""
            echo "  🧬  RFCellClassificator v2.0 — dev shell"
            echo "  Python: $(python --version)"
            echo "  BLAS:   openblas (ottimizzato)"
            echo ""
            # Installa dipendenze non in nixpkgs via pip (scanpy, leidenalg)
            if ! python -c "import scanpy" 2>/dev/null; then
              echo "  [setup] installazione scanpy + leidenalg..."
              pip install --quiet scanpy leidenalg
            fi
            export OPENBLAS_NUM_THREADS=4   # Pi 5 ha 4 core
            export OMP_NUM_THREADS=4
          '';

          # Variabili per linkare openblas a numpy/scipy
          LD_LIBRARY_PATH = "${pkgs.openblas}/lib";
        };

        # ── Package eseguibile: `nix run` ───────────────────────────────
        packages.default = pkgs.writeShellScriptBin "rfcell" ''
          cd ${self}
          exec ${pythonEnv}/bin/python random_forest_tui.py "$@"
        '';
      }
    );
}

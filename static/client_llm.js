/**
 * CommitCon Sherpa Driver Copilot — In-Browser Hardware-Accelerated LLM Engine
 * Powered by Wllama (WebAssembly + SIMD llama.cpp runtime).
 * Executes the REAL fine-tuned Qwen 0.8B Q4_K_M GGUF model directly on user hardware.
 * Modeled on the LPU Supervised Project client engine architecture.
 * Zero external cloud API calls. Zero GPU quota limits. Zero server RAM overhead.
 */

class ClientSherpaEngine {
    constructor() {
        this.wllama = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.loadProgress = 0;
        this.hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
        this.activeProvider = 'WASM SIMD Hardware Acceleration';
        this.modelUrl = "https://huggingface.co/spaces/Mohd-Shujath/commitcon-driver-inference/resolve/main/Qwen3.5-0.8B.Q4_K_M.gguf";
    }

    /**
     * Inspects host device hardware capabilities.
     */
    getHardwareInfo() {
        return {
            mode: this.hasWebGPU ? 'Local WebGPU / WASM Hardware Engine' : 'Local WASM SIMD Hardware Engine',
            shortMode: 'Local Hardware (WASM/GPU)',
            badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
            icon: '⚡',
        };
    }

    /**
     * Lazy-loads the fine-tuned Qwen 0.8B GGUF model into local client memory.
     */
    async loadModel(onProgress = null) {
        if (this.isLoaded) return true;
        if (this.isLoading) {
            while (this.isLoading) {
                await new Promise(r => setTimeout(r, 200));
            }
            return this.isLoaded;
        }

        this.isLoading = true;
        try {
            if (onProgress) onProgress(5, "Initializing local WebAssembly llama runtime...");

            // Dynamically import Wllama and CDN paths
            const { Wllama } = await import("https://cdn.jsdelivr.net/npm/@wllama/wllama@latest/esm/index.js");
            let configPaths;
            try {
                const cdnMod = await import("https://cdn.jsdelivr.net/npm/@wllama/wllama@latest/esm/wasm-from-cdn.js");
                configPaths = cdnMod.default || cdnMod.WasmFromCDN;
            } catch (cdnErr) {
                const baseURL = "https://cdn.jsdelivr.net/npm/@wllama/wllama@latest/esm/";
                configPaths = {
                    "single-thread/wllama.js": baseURL + "single-thread/wllama.js",
                    "single-thread/wllama.wasm": baseURL + "single-thread/wllama.wasm",
                    "multi-thread/wllama.js": baseURL + "multi-thread/wllama.js",
                    "multi-thread/wllama.wasm": baseURL + "multi-thread/wllama.wasm",
                    "multi-thread/wllama.worker.mjs": baseURL + "multi-thread/wllama.worker.mjs",
                };
            }

            this.wllama = new Wllama(configPaths);

            if (onProgress) onProgress(15, "Loading fine-tuned Qwen 0.8B GGUF into hardware cache...");

            // Try local server endpoint first, fallback to CDN
            let targetUrl = "/model/qwen.gguf";
            try {
                const headCheck = await fetch(targetUrl, { method: "HEAD" });
                if (!headCheck.ok) targetUrl = this.modelUrl;
            } catch {
                targetUrl = this.modelUrl;
            }

            await this.wllama.loadModelFromUrl(targetUrl, {
                n_ctx: 2048,
                n_gpu_layers: this.hasWebGPU ? 32 : 0,
                progressCallback: ({ loaded, total }) => {
                    const pct = total > 0 ? Math.round((loaded / total) * 100) : 50;
                    this.loadProgress = pct;
                    if (onProgress) onProgress(pct, `Loading fine-tuned Qwen 0.8B: ${pct}%`);
                }
            });

            this.isLoaded = true;
            this.isLoading = false;
            if (onProgress) onProgress(100, "Fine-tuned Qwen 0.8B loaded on local hardware!");
            return true;
        } catch (err) {
            console.error("Local hardware model load error:", err);
            this.isLoading = false;
            return false;
        }
    }

    /**
     * Executes inference with the REAL fine-tuned neural network on the driver's prompt.
     */
    async ask(userPrompt, driverZoneId, cityRecommendations, onProgress = null) {
        const t0 = performance.now();

        // 1. Ensure the real fine-tuned model is loaded into local memory
        if (!this.isLoaded) {
            const ok = await this.loadModel(onProgress);
            if (!ok) {
                throw new Error("Could not initialize local model on this device.");
            }
        }

        // 2. Prepare live driver repositioning context
        const zones = (cityRecommendations && cityRecommendations.zones) ? cityRecommendations.zones : [];
        const topRecs = [...zones]
            .filter(z => (z.supply_gap && z.supply_gap > 0) || z.recommended_action === 'REPOSITION_IN')
            .sort((a, b) => (b.supply_gap || 0) - (a.supply_gap || 0))
            .slice(0, 3);

        const recSummary = topRecs.map(r => `${r.zone_id} ${r.zone_name} (gap: +${r.supply_gap}, ETA: ${r.travel_time_min || 15}m)`).join(", ");

        const systemPrompt = `You are Sherpa, an AI driving assistant for the CommitCon ride-hailing repositioning platform in Bengaluru. Provide concise, direct, helpful recommendations to drivers about where to reposition for high demand, earnings, and low traffic delays.
Live Context: Driver is in Zone ${driverZoneId}. Top citywide demand shortage zones right now: ${recSummary || 'Z02 Airport (+10 drivers), Z07 Railway Station (+7 drivers)'}.`;

        // 3. Execute inference using the REAL fine-tuned Qwen 0.8B neural network
        const response = await this.wllama.createChatCompletion({
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt }
            ],
            temperature: 0.7,
            max_tokens: 256,
        });

        const answer = response.choices[0].message.content.trim();
        const latencyMs = Math.round(performance.now() - t0);

        return {
            status: "success",
            answer: answer,
            latencyMs: latencyMs,
            model: "qwen-0.8b-gguf-hardware"
        };
    }
}

// Instantiate globally on client window
if (typeof window !== 'undefined') {
    window.clientSherpa = new ClientSherpaEngine();
}

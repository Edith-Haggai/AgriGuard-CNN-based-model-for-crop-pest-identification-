"use client";

import { useRef, useState } from "react";

export default function UploadPage() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "processing" | "done">("idle");

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    setStatus("idle");
  }

  function analyze() {
    setStatus("processing");
    setTimeout(() => setStatus("done"), 1200);
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
      <h1 className="text-2xl font-semibold text-slate-900">Upload Image for Prediction</h1>
      <p className="mt-2 text-sm text-slate-600">
        Upload an image of your crops to detect pests and diseases using AI.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        {/* Uploader */}
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
          <div
            className="flex h-48 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-slate-300 bg-slate-50"
            onClick={() => inputRef.current?.click()}
          >
            <div className="text-sm text-slate-600">Drop your image here</div>
            <button className="mt-3 inline-flex h-9 items-center rounded-md bg-slate-900 px-4 text-sm font-medium text-white">
              Choose File
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onFileChange}
            />
          </div>

          {/* Preview */}
          <div className="mt-6">
            <div className="text-sm font-medium text-slate-900">Image Preview</div>
            <div className="mt-2 flex h-56 items-center justify-center rounded-md border border-slate-200 bg-slate-50">
              {preview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={preview} alt="preview" className="h-full w-full rounded-md object-contain" />
              ) : (
                <div className="text-sm text-slate-500">No image selected</div>
              )}
            </div>
          </div>

          <div className="mt-6">
            <button
              className="inline-flex h-10 items-center rounded-md bg-slate-900 px-5 text-white disabled:cursor-not-allowed disabled:opacity-50"
              disabled={!preview || status === "processing"}
              onClick={analyze}
            >
              {status === "processing" ? "Analyzing..." : "Analyze Image"}
            </button>
          </div>
        </div>

        {/* Side results */}
        <aside className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-900">Prediction Results</h2>
          <ul className="mt-4 space-y-3 text-sm">
            <li className="flex items-center justify-between">
              <span>Disease Detected</span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700">
                {status === "done" ? "Leaf Spot" : "Pending"}
              </span>
            </li>
            <li className="flex items-center justify-between">
              <span>Confidence</span>
              <span className="text-slate-700">{status === "done" ? "93%" : "—"}</span>
            </li>
            <li>
              <div className="text-slate-900">Recommended Actions</div>
              <p className="mt-1 text-xs text-slate-600">
                {status === "done" ? "Apply bio‑fungicide and remove affected leaves." : "—"}
              </p>
            </li>
          </ul>

          <div className="mt-8">
            <div className="text-sm font-semibold text-slate-900">Analysis History</div>
            <ul className="mt-3 space-y-2 text-xs text-slate-600">
              <li>Tomato Leaf Sample — 2 hours ago</li>
              <li>Corn Pest Check — 1 day ago</li>
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}



export default function AdminDashboard() {
  return (
    <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
      <h1 className="text-2xl font-semibold text-slate-900">Admin Dashboard</h1>
      <p className="mt-1 text-sm text-slate-600">Overview of platform activity and recent events.</p>

      {/* Stat cards */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
        {[
          { label: "Detections Today", value: 128 },
          { label: "Models Deployed", value: 4 },
          { label: "Active Users", value: 2_394 },
          { label: "Avg. Inference", value: "2.8s" },
          { label: "Data Ingested", value: "12.7 GB" },
          { label: "Issues Open", value: 5 },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="text-xs text-slate-500">{s.label}</div>
            <div className="mt-2 text-2xl font-semibold text-slate-900">{s.value}</div>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="text-sm font-medium text-slate-900">Disease Distribution</div>
          <div className="mt-4 grid h-56 place-items-center rounded-md bg-slate-50 text-xs text-slate-500">
            Pie Chart Placeholder
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="text-sm font-medium text-slate-900">Detection Trends</div>
          <div className="mt-4 grid h-56 place-items-center rounded-md bg-slate-50 text-xs text-slate-500">
            Line Chart Placeholder
          </div>
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="text-sm font-medium text-slate-900">Recent Activity</div>
        <ul className="mt-4 space-y-3 text-sm text-slate-700">
          <li>User Alice uploaded 200 new images</li>
          <li>Model v1.2 deployed to production</li>
          <li>John approved access for researcher Mark</li>
        </ul>
      </div>
    </div>
  );
}



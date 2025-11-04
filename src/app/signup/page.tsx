export default function SignUpPage() {
  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <h1 className="text-center text-2xl font-semibold text-slate-900">Create Account</h1>
        <p className="mt-1 text-center text-sm text-slate-600">Join AgriGuard to protect your crops</p>

        <div className="mt-6 space-y-4">
          <div>
            <label className="text-sm font-medium text-slate-900">Full Name</label>
            <input className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your full name" />
          </div>
          <div>
            <label className="text-sm font-medium text-slate-900">Email Address</label>
            <input type="email" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your email" />
          </div>
          <div>
            <label className="text-sm font-medium text-slate-900">Password</label>
            <input type="password" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Create a password" />
          </div>
          <div>
            <label className="text-sm font-medium text-slate-900">Confirm Password</label>
            <input type="password" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Confirm your password" />
          </div>
          <div>
            <label className="text-sm font-medium text-slate-900">Role</label>
            <select className="mt-1 h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm">
              <option>Farmer</option>
              <option>Researcher</option>
              <option>Admin</option>
            </select>
          </div>
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-700">
            <input type="checkbox" /> I agree to the Terms of Service and Privacy Policy
          </label>
          <button className="mt-2 inline-flex h-10 w-full items-center justify-center rounded-md bg-slate-900 px-5 text-white">Create Account</button>
        </div>

        <div className="mt-4 text-center text-sm text-slate-700">
          Already have an account? <a href="/signin" className="text-slate-900 underline">Sign in</a>
        </div>
      </div>
    </div>
  );
}



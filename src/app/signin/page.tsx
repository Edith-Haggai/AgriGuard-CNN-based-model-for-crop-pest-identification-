export default function SignInPage() {
  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <h1 className="text-center text-2xl font-semibold text-slate-900">AgriGuard</h1>
        <p className="mt-1 text-center text-sm text-slate-600">Sign in to your account</p>

        <div className="mt-6 space-y-4">
          <div>
            <label className="text-sm font-medium text-slate-900">Email</label>
            <input type="email" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your email" />
          </div>
          <div>
            <label className="text-sm font-medium text-slate-900">Password</label>
            <input type="password" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your password" />
          </div>
          <button className="mt-2 inline-flex h-10 w-full items-center justify-center rounded-md bg-slate-900 px-5 text-white">Sign In</button>
        </div>

        <div className="mt-6 text-center text-xs text-slate-500">Or sign in with</div>
        <div className="mt-3 space-y-2">
          {['Google','Apple','GitHub'].map((p)=> (
            <button key={p} className="inline-flex h-10 w-full items-center justify-center rounded-md border border-slate-300 bg-white px-5 text-sm text-slate-700 hover:bg-slate-50">Continue with {p}</button>
          ))}
        </div>

        <div className="mt-4 text-center text-sm text-slate-700">
          <a href="#" className="text-slate-900 underline">Forgot Password?</a>
        </div>
        <div className="mt-2 text-center text-sm text-slate-700">
          Don&apos;t have an account? <a href="/signup" className="text-slate-900 underline">Create Account</a>
        </div>
      </div>
    </div>
  );
}



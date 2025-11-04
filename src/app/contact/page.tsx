export default function ContactPage() {
  return (
    <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
      <h1 className="text-2xl font-semibold text-slate-900">Contact Us</h1>
      <p className="mt-1 text-sm text-slate-600">Get in touch with our team. We're here to help.</p>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        {/* Form */}
        <form className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="text-sm font-medium text-slate-900">Full Name</label>
              <input className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your full name" />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-900">Email Address</label>
              <input type="email" className="mt-1 h-10 w-full rounded-md border border-slate-300 px-3 text-sm" placeholder="Enter your email" />
            </div>
          </div>
          <div className="mt-4">
            <label className="text-sm font-medium text-slate-900">Message</label>
            <textarea className="mt-1 h-32 w-full rounded-md border border-slate-300 px-3 py-2 text-sm" placeholder="Tell us how we can help you" />
          </div>
          <button className="mt-6 inline-flex h-10 items-center rounded-md bg-slate-900 px-5 text-white">Send Message</button>
        </form>

        {/* Info */}
        <aside className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="text-sm font-semibold text-slate-900">Get in touch</div>
          <ul className="mt-3 space-y-2 text-sm text-slate-700">
            <li>Phone: +1 (555) 123-4567</li>
            <li>Email: support@agriguard.com</li>
            <li>Address: 123 Agriculture Drive, Farm Valley, CA 94005</li>
          </ul>
          <div className="mt-6 h-40 w-full rounded-md bg-slate-100" />
          <div className="mt-6 rounded-lg bg-slate-50 p-4 text-center text-xs text-slate-600">
            Office Hours: Mon–Fri 9:00 AM – 6:00 PM PST, Sat–Sun 10:00 AM – 4:00 PM PST
          </div>
        </aside>
      </div>
    </div>
  );
}



# The benchmark report: a static site with a front page of the machines, the summary table and
# Dolan-Moré performance profiles, a page per set with every instance, and the result files
# and solver logs under `data/`.

const COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

const SET_TITLES = Dict("acopf" => "ACOPF", "cops" => "COPS", "cutest" => "CUTEst", "mittelmann" => "Mittelmann")

const BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

const STYLE = """
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body,table,input,.form-control{font-size:14px;background:#fcfcfb}
h1,h2,h3{font-size:14px;font-weight:600}h1{margin-top:1em}h2{margin:1.8em 0 .4em}h3{margin:1.2em 0 .4em;color:#52514e}
a{text-decoration:none}a:hover{text-decoration:underline}
table{font-variant-numeric:tabular-nums;width:auto!important}
td,th{text-align:right;white-space:nowrap}td:first-child,th:first-child,td.l,th.l{text-align:left}
th{color:#52514e;font-weight:600}th.s{cursor:pointer}
.m{color:#52514e}.fail{font-weight:600}svg text{font-size:14px;fill:#52514e}
svg{display:block;width:100%;max-width:620px;height:auto}
.key{display:inline-block;margin-right:1.2em}.key i{display:inline-block;width:14px;height:3px;border-radius:2px;margin:0 6px 3px 0}
.prof{position:relative;display:inline-block;vertical-align:top;width:100%;max-width:620px;margin:0 1.5em 1em 0}
.tip{position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #d6d5cf;border-radius:4px;padding:4px 8px;white-space:nowrap}
.bar{background:#fcfcfb;border-bottom:1px solid #d6d5cf;padding:.45em 0}
.bar .top{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar nav{display:flex;gap:1.1em;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch}
table.summary{min-width:26em}table.summary tr.set td{background:#f3f2ee;font-weight:600}table.summary tr.set .m{font-weight:400;margin-left:.6em}
"""

const SCRIPT = """
function bind(){
document.querySelectorAll('.prof').forEach(p=>{const s=p.querySelector('svg'),t=p.querySelector('.tip'),l=p.querySelector('.x');
const d=JSON.parse(s.dataset.c);
s.addEventListener('mousemove',e=>{const r=s.getBoundingClientRect(),x=(e.clientX-r.left)*d.w/r.width;
if(x<d.l||x>d.r){t.style.display='none';l.style.display='none';return}
const tau=Math.pow(2,(x-d.l)/(d.r-d.l)*d.x);
t.innerHTML='within '+tau.toFixed(2)+'× of the best<br>'+d.s.map(c=>c[0]+': '+(100*c[1].filter(v=>v<=tau).length/d.n).toFixed(1)+'%').join('<br>');
t.style.display='block';t.style.left=Math.max(0,Math.min(e.clientX-r.left+12,r.width-200))+'px';t.style.top='8px';
l.setAttribute('x1',x);l.setAttribute('x2',x);l.style.display='block'});
s.addEventListener('mouseleave',()=>{t.style.display='none';l.style.display='none'})});
const q=document.querySelector('#filter');
document.querySelectorAll('table.cases').forEach(tb=>{const body=tb.tBodies[0];
q.addEventListener('input',e=>{const v=e.target.value.toLowerCase();
for(const r of body.rows)r.style.display=r.cells[0].textContent.toLowerCase().includes(v)?'':'none'});
tb.querySelectorAll('th.s').forEach(h=>{let up=true;h.addEventListener('click',()=>{const i=h.cellIndex;up=!up;
const key=r=>{const v=r.cells[i].textContent,n=parseFloat(v);return isNaN(n)?v:n};
[...body.rows].sort((a,b)=>{const x=key(a),y=key(b),c=x<y?-1:x>y?1:0;return up?-c:c}).forEach(r=>body.append(r))})})})
}
bind();
// while a run is going, take the page again every few seconds and swap its content in place
if(document.querySelector('meta[name=live]'))setInterval(async()=>{
const r=await fetch(location.href,{cache:'no-store'});if(!r.ok)return;
const d=new DOMParser().parseFromString(await r.text(),'text/html'),q=document.querySelector('#filter')?.value;
document.querySelector('#live').replaceWith(d.querySelector('#live'));bind();
if(q){const f=document.querySelector('#filter');f.value=q;f.dispatchEvent(new Event('input'))}
if(!d.querySelector('meta[name=live]'))document.querySelector('meta[name=live]').remove()},5000);
"""

escape_html(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function status_name(code)
    code == KILLED_STATUS && return "killed at the limit"
    code == ERROR_STATUS && return "error"
    code == Int(MadNLP.SOLVE_SUCCEEDED) && return "solved"
    code == Int(MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL) && return "solved to acceptable level"
    return try
        lowercase(replace(string(MadNLP.Status(Int(code))), "_" => " "))
    catch
        string(code)
    end
end

# The letter of an outcome in the instance tables, as in Hans Mittelmann's tables.
function status_letter(code)
    code == Int(MadNLP.SOLVE_SUCCEEDED) && return "ok"
    code == Int(MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL) && return "a"
    (code == KILLED_STATUS || code == Int(MadNLP.MAXIMUM_WALLTIME_EXCEEDED)) && return "t"
    code == Int(MadNLP.MAXIMUM_ITERATIONS_EXCEEDED) && return "i"
    return "f"
end

# The columns of a report: one per solver and configuration of a side.
struct Side
    label::String
    color::String
    key::Tuple{String, String}   # device, configuration
    base::Bool
end

# Per device the columns are Ipopt, master and the current results.
function report_sides(cand, base)
    suffix(key) = key[2] == "default" ? "" : " " * key[2]
    label(key, name) = key[2] == "ipopt" ? "Ipopt" : name * suffix(key)
    keys_of(rs) = rs === nothing ? [] : sort!(unique!([(k[1], k[3]) for k in keys(rs.rows)]); by = k -> (k[1] != "cpu", k))
    sides = [(label(k, "master"), k, true) for k in keys_of(base) if k[2] != "ipopt"]
    append!(sides, [(label(k, "current"), k, false) for k in keys_of(cand)])
    sort!(sides; by = s -> s[2][2] != "ipopt")
    return [Side(l, COLORS[mod1(i, length(COLORS))], k, b) for (i, (l, k, b)) in enumerate(sides)]
end

side_rows(side, cand, base, set) = get((side.base ? base : cand).rows, (side.key[1], set, side.key[2]), Dict{String, Any}())

# What a performance profile compares, each on its own.
const PROFILE_METRICS = ["Wall time" => (r -> max(r.t_total, 1.0e-3)), "Iterations" => (r -> max(r.iters, 1))]

# Per side the ratios to the best side on the instances that every side has a result for; a
# failure has no ratio.
function profile(rows, value)
    cases = isempty(rows) ? Set{String}() : reduce(intersect, [Set(keys(r)) for r in rows])
    ratios = [Float64[] for _ in rows]
    for case in cases
        times = [solved(r[case]) ? value(r[case]) : Inf for r in rows]
        best = minimum(times)
        isfinite(best) || continue
        for (i, t) in enumerate(times)
            isfinite(t) && push!(ratios[i], t / best)
        end
    end
    return length(cases), sort!.(ratios)
end

function profile_svg(io, title, sides, n, ratios)
    w, h, left, right, top, bottom = 620, 330, 44, 150, 12, 36
    xmax = clamp(ceil(log2(maximum(Iterators.flatten(ratios); init = 2.0))), 1, 12)
    X(t) = left + (w - left - right) * min(log2(t), xmax) / xmax
    Y(f) = top + (h - top - bottom) * (1 - f)
    json = join(["[\"$(s.label)\",[$(join(round.(r; digits = 3), ","))]]" for (s, r) in zip(sides, ratios)], ",")
    print(io, "<div class=\"tip\"></div><svg viewBox=\"0 0 $w $h\" role=\"img\" aria-label=\"$(escape_html(title))\" ")
    print(io, "data-c='{\"w\":$w,\"l\":$left,\"r\":$(w - right),\"x\":$xmax,\"n\":$n,\"s\":[$json]}'>")
    for f in 0:0.25:1
        print(io, "<line x1=\"$left\" x2=\"$(w - right)\" y1=\"$(Y(f))\" y2=\"$(Y(f))\" stroke=\"#e4e3df\"/>")
        print(io, "<text x=\"$(left - 6)\" y=\"$(Y(f) + 4)\" text-anchor=\"end\">$(round(Int, 100f))%</text>")
    end
    step = ceil(Int, xmax / 8)
    for k in 0:step:Int(xmax)
        print(io, "<text x=\"$(X(2.0^k))\" y=\"$(h - bottom + 16)\" text-anchor=\"middle\">$(2^k)×</text>")
    end
    print(io, "<text x=\"$((left + w - right) / 2)\" y=\"$(h - 4)\" text-anchor=\"middle\">within this factor of the best</text>")
    print(io, "<line class=\"x\" y1=\"$top\" y2=\"$(h - bottom)\" stroke=\"#8a8983\" style=\"display:none\"/>")
    ends = Float64[]
    for (s, r) in zip(sides, ratios)
        d = IOBuffer()
        print(d, "M$(X(1.0)) $(Y(0.0))")
        for (i, t) in enumerate(r)
            print(d, "L$(X(t)) $(Y((i - 1) / n))L$(X(t)) $(Y(i / n))")
        end
        print(d, "L$(w - right) $(Y(length(r) / n))")
        print(io, "<path d=\"$(String(take!(d)))\" fill=\"none\" stroke=\"$(s.color)\" stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>")
        push!(ends, Y(length(r) / n))
    end
    # labels at the ends of the curves, kept apart
    order = sortperm(ends)
    y = -Inf
    for i in order
        y = max(ends[i] + 4, y + 15)
        print(io, "<text x=\"$(w - right + 6)\" y=\"$y\">$(escape_html(sides[i].label))</text>")
    end
    println(io, "</svg>")
    return
end

# The solvers shown for a device: its own, with Ipopt as the reference of every device. MadNLP on
# the host and a solver on a GPU are never compared.
on_device(sides, device) = [s for s in sides if s.key[1] == device || s.key[2] == "ipopt"]

# One profile per device.
function profile_section(io, set, sides, cand, base, wall)
    for device in unique!([s.key[1] for s in sides])
        device_profile(io, set, device, on_device(sides, device), cand, base, wall)
    end
    return
end

function device_profile(io, set, device, sides, cand, base, wall)
    rows = [side_rows(s, cand, base, set) for s in sides]
    kept = [i for i in eachindex(sides) if !isempty(rows[i])]
    isempty(kept) && return
    println(io, "<h3>", device == "cuda" ? "GPU" : "CPU", "</h3>")
    if length(kept) < 2
        println(io, "<p class=\"m\">No profile: there is one solver on this device, and no archived results to compare with.</p>")
        return
    end
    n = length(reduce(intersect, [Set(keys(r)) for r in rows[kept]]))
    println(io, "<p class=\"m\">Over the $n instances that all of them have a result for. A curve gives the share of them solved within a factor of the best.</p>")
    println(io, "<p>", join(["<span class=\"key\"><i style=\"background:$(s.color)\"></i>$(escape_html(s.label))</span>" for s in sides[kept]]), "</p>")
    n > 0 || return
    for (name, value) in PROFILE_METRICS
        _, ratios = profile(rows[kept], value)
        println(io, "<div class=\"prof\"><p><b>", name, "</b></p>")
        profile_svg(io, "$name profile of $set on $device", sides[kept], n, ratios)
        println(io, "</div>")
    end
    return
end

function page(path, title, content, bar; refresh = false)
    tmp = path * ".tmp"
    write(
        tmp, "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        refresh ? "<meta name=\"live\" content=\"1\">" : "",
        "<title>", escape_html(title), "</title><link rel=\"stylesheet\" href=\"", BOOTSTRAP, "\"><style>", STYLE, "</style><body><div id=\"live\">",
        bar, "<main class=\"container\">", content, "</main></div><script>", SCRIPT, "</script></body></html>\n",
    )
    chmod(tmp, 0o644)
    mv(tmp, path; force = true)
    return
end

# The head of a table of Hans Mittelmann's benchmarks, per set: the scaled shifted geometric mean,
# with the reference solver at 1, and the instances solved, over the instances all solvers have run.
function summary_table(io, sets, sides, cand, base, wall)
    kept = [s for s in sides if any(set -> !isempty(side_rows(s, cand, base, set)), sets)]
    sort!(kept; by = s -> s.key[2] != "ipopt")
    println(io, "<div class=\"table-responsive\"><table class=\"table table-sm summary\"><thead><tr><th></th>")
    foreach(s -> print(io, "<th>", escape_html(s.label), "</th>"), kept)
    println(io, "</tr></thead><tbody><tr class=\"m\"><td>rev</td>")
    foreach(s -> print(io, "<td>", escape_html(get((s.base ? base : cand).revs, s.key, "–")), "</td>"), kept)
    println(io, "</tr><tr class=\"m\"><td>benchmark</td>")
    foreach(s -> print(io, "<td>", escape_html(get((s.base ? base : cand).harness, s.key, "–")), "</td>"), kept)
    println(io, "</tr>")
    blank = "<td></td>"
    for set in sets
        rows = [side_rows(s, cand, base, set) for s in kept]
        have = findall(!isempty, rows)
        isempty(have) && continue
        cases = collect(reduce(intersect, [Set(keys(rows[i])) for i in have]))
        isempty(cases) && continue
        means = Dict(i => sgm([score_time(rows[i][c], wall) for c in cases]) + 10 for i in have)
        # Ipopt is the reference where it has run, else the best solver
        reference = findfirst(i -> kept[i].key[2] == "ipopt", have)
        scale = reference === nothing ? minimum(values(means)) : means[have[reference]]
        cell(f) = join([i in have ? "<td>" * f(i) * "</td>" : blank for i in eachindex(kept)])
        println(io, "<tr class=\"set\"><td colspan=\"", length(kept) + 1, "\"><a href=\"$set.html\">$set</a> <span class=\"m\">$(length(cases)) problems</span></td></tr>")
        println(io, "<tr><td>solved</td>", cell(i -> "$(count(c -> solved(rows[i][c]), cases))/$(length(cases))"), "</tr>")
        println(io, "<tr><td>SGM (s)</td>", cell(i -> @sprintf("%.2f", means[i] - 10)), "</tr>")
        println(io, "<tr><td>SSGM</td>", cell(i -> @sprintf("%.3g", means[i] / scale)), "</tr>")
    end
    println(io, "</tbody></table></div>")
    return
end

# Where the time goes, over the instances with a breakdown. Ipopt reports none.
function breakdown(io, sides, rows)
    println(io, "<div class=\"table-responsive\"><table class=\"table table-sm\"><thead><tr><th>solver</th><th>total (s)</th><th>linear solver</th><th>evaluations</th><th>initialization</th><th>other</th></tr></thead><tbody>")
    for (s, r) in zip(sides, rows)
        known = [x for x in values(r) if isfinite(x.t_linsolve) && x.t_total > 0]
        isempty(known) && continue
        total = sum(x -> x.t_total, known)
        share(f) = 100 * sum(f, known) / total
        lin, eval, init = share(x -> x.t_linsolve), share(x -> x.t_eval), share(x -> x.t_init)
        @printf(io, "<tr><td>%s</td><td>%.1f</td><td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td></tr>\n", escape_html(s.label), total, lin, eval, init, 100 - lin - eval - init)
    end
    println(io, "</tbody></table></div>")
    return
end

function instance_table(io, set, sides, rows, logs)
    println(io, "<div class=\"table-responsive\"><table class=\"table table-sm table-hover cases\"><thead><tr><th class=\"s\" rowspan=\"2\">instance</th><th class=\"s\" rowspan=\"2\">nv</th><th class=\"s\" rowspan=\"2\">nc</th>")
    foreach(s -> print(io, "<th colspan=\"12\" class=\"l\">", escape_html(s.label), "</th>"), sides)
    print(io, "</tr><tr>")
    foreach(_ -> print(io, "<th class=\"s l\">st</th><th class=\"s\">it</th><th class=\"s\">t (s)</th><th class=\"s\">obj</th><th class=\"s\">inf</th><th class=\"s\">nfac</th><th class=\"s\">nsol</th><th class=\"s\">nf</th><th class=\"s\">ng</th><th class=\"s\">nc</th><th class=\"s\">nj</th><th class=\"s\">nh</th>"), sides)
    println(io, "</tr></thead><tbody>")
    for case in sort!(collect(reduce(union, [Set(keys(r)) for r in rows]; init = Set{String}())))
        any_row = first(r[case] for r in rows if haskey(r, case))
        print(io, "<tr><td>", escape_html(case), "</td><td>", any_row.nvar, "</td><td>", any_row.ncon, "</td>")
        for (s, r) in zip(sides, rows)
            if !haskey(r, case)
                print(io, "<td class=\"l\">–</td>", "<td></td>"^11)
                continue
            end
            row = r[case]
            link = s.base ? nothing : get(logs, (s.key[1], set, case, s.key[2]), nothing)
            print(io, "<td class=\"l", solved(row) ? "" : " fail", "\" title=\"", status_name(row.status), "\">", link === nothing ? status_letter(row.status) : "<a href=\"$link\">$(status_letter(row.status))</a>", "</td><td>", row.iters, "</td>")
            @printf(io, "<td>%.3f</td><td>%.6g</td><td>%.1e</td>", row.t_total, row.obj, row.primal_feas)
            foreach(n -> print(io, "<td>", n, "</td>"), (row.n_fact, row.n_backsolve, row.n_obj, row.n_grad, row.n_con, row.n_jac, row.n_hess))
        end
        println(io, "</tr>")
    end
    println(io, "</tbody></table></div>")
    return
end

# What every worker is on.
function workers_page(site, root, bar)
    io = IOBuffer()
    println(io, "<h1>Workers</h1>")
    for dir in sort!(readdir(root; join = true))
        file = joinpath(dir, "summary.md")
        isfile(file) || continue
        m = match(r"^```\n(.*?)\n```"s, read(file, String))
        m === nothing && continue
        println(io, "<h2>", escape_html(basename(dir)), "</h2>")
        workers = String[]
        for line in split(m[1], '\n')
            w = match(r"worker (\d+): (\S+) (\S*) for (\d+) s, ([\d.]+) GB", line)
            if w === nothing
                println(io, "<p class=\"m\">", escape_html(strip(line)), "</p>")
            else
                push!(workers, "<tr><td>$(w[1])</td><td class=\"l\">$(w[2])</td><td class=\"l\">$(escape_html(w[3]))</td><td>$(w[4])</td><td>$(w[5])</td></tr>")
            end
        end
        isempty(workers) || println(io, "<div class=\"table-responsive\"><table class=\"table table-sm\"><thead><tr><th>worker</th><th class=\"l\">state</th><th class=\"l\">instance</th><th>for (s)</th><th>memory (GB)</th></tr></thead><tbody>", join(workers), "</tbody></table></div>")
    end
    page(joinpath(site, "workers.html"), "Workers", String(take!(io)), bar; refresh = true)
    return
end

function set_page(site, set, sides, cand, base, wall, logs, bar; refresh)
    rows = [side_rows(s, cand, base, set) for s in sides]
    keep = [i for i in eachindex(sides) if !isempty(rows[i])]
    sides, rows = sides[keep], rows[keep]
    io = IOBuffer()
    println(io, "<h1>", escape_html(set), "</h1>")
    println(io, "<h2>Performance profile</h2>")
    profile_section(io, set, sides, cand, base, wall)
    println(io, "<h2>Time breakdown</h2>")
    for device in unique!([s.key[1] for s in sides])
        here = [i for i in eachindex(sides) if sides[i].key[1] == device]
        println(io, "<h3>", device == "cuda" ? "GPU" : "CPU", "</h3>")
        breakdown(io, sides[here], rows[here])
    end
    println(io, "<h2>Instances</h2><p class=\"m\">nv, nc: variables, constraints. st: ok solved, a solved to the acceptable level, t time limit, i iteration limit, f other failure; click it for the solver log. it: iterations. inf: infeasibility. nfac, nsol: factorizations, linear solves. nf, ng, nc, nj, nh: evaluations of the objective, gradient, constraints, Jacobian, Hessian.</p><input id=\"filter\" class=\"form-control form-control-sm mb-2\" style=\"max-width:20em\" placeholder=\"Filter by name\">")
    for device in unique!([s.key[1] for s in sides])
        here = [i for i in eachindex(sides) if sides[i].key[1] == device]
        println(io, "<h3>", device == "cuda" ? "GPU" : "CPU", "</h3>")
        instance_table(io, set, sides[here], rows[here], logs)
    end
    page(joinpath(site, "$set.html"), "$set: MadNLP benchmark", String(take!(io)), bar; refresh)
    return
end

# Bring `data/` up to date with the result directories, and index the solver logs.
function sync_data(site, root)
    logs = Dict{Tuple{String, String, String, String}, String}()
    for dir in sort!(readdir(root))
        source = joinpath(root, dir)
        isdir(source) || continue
        device = nothing
        for (path, _, files) in walkdir(source), file in files
            endswith(file, ".tmp") && continue
            from = joinpath(path, file)
            relative = relpath(from, root)
            to = joinpath(site, "data", relative)
            if !isfile(to) || mtime(to) < mtime(from)
                mkpath(dirname(to))
                # a solver log carries terminal colours
                endswith(file, ".txt") ? write(to, replace(read(from, String), r"\e\[[0-9;]*m" => "")) : cp(from, to; force = true)
                chmod(to, 0o644)
            end
            endswith(file, ".meta.toml") && device === nothing && (device = read_meta(from[1:(end - 10)])["device"])
        end
        logdir = joinpath(source, "logs")
        (device === nothing || !isdir(logdir)) && continue
        for set in readdir(logdir), file in readdir(joinpath(logdir, set))
            m = match(r"^(.*)__(.*)\.txt$", file)
            m === nothing || (logs[(device, set, m[1], m[2])] = "data/$dir/logs/$set/$file")
        end
    end
    return logs
end

"""
    write_report(site, root; base = nothing, wall = 7200.0, title, link = nothing, reference = nothing, finished = false)

Write the report of the result directories under `root` to the directory `site`. `base` is a
`ResultSet` to compare with, `reference` the Ipopt reference and `link` a page to link at the
top, such as the pull request.
While `finished = false` the pages update themselves in place and the front page shows what
every worker is on.
"""
function write_report(site, root; base = nothing, wall = 7200.0, title = "MadNLP benchmark", link = nothing, reference = nothing, finished::Bool = false)
    mkpath(site)
    chmod(site, 0o755)
    logs = sync_data(site, root)
    cand = from_directory(root, "this run")
    if reference !== nothing
        merge!(cand.rows, reference.rows)
        merge!(cand.revs, reference.revs)
        merge!(cand.harness, reference.harness)
    end
    sides = report_sides(cand, base)
    sets = sort!(unique!([k[2] for k in keys(cand.rows)]))
    stamp = Dates.format(now(UTC), "yyyy-mm-dd HH:MM:SS")
    heading = (finished ? "Finished" : "In progress") * ", updated $stamp UTC."
    devices = unique!([s.key[1] for s in sides])
    # the bar pinned to the top of every page
    families = ["<a href=\"index.html\">Summary</a>"; ["<a href=\"$set.html\">$(get(SET_TITLES, set, set))</a>" for set in sets]]
    tools = String[]
    finished || push!(tools, "<a href=\"workers.html\">Workers</a>")
    push!(tools, "<a href=\"data/\">Files</a>")
    link === nothing || push!(tools, "<a href=\"" * escape_html(link) * "\">Pull request</a>")
    bar = "<div class=\"bar sticky-top\"><div class=\"container\"><div class=\"top\"><b>" * escape_html(title) * "</b> <span class=\"m\">" * heading *
        "</span></div><nav>" * join(families) * "</nav><nav>" * join(tools) * "</nav></div></div>"
    for set in sets
        set_page(site, set, sides, cand, base, wall, logs, bar; refresh = !finished)
    end
    finished || workers_page(site, root, bar)
    io = IOBuffer()
    for device in devices
        println(io, "<h2>", solver_name(device), device == "cuda" ? " (GPU)" : " (CPU)", "</h2>")
        machines = get(cand.machines, device, String[])
        isempty(machines) || println(io, "<p class=\"m\">", escape_html(join(machines, "; ")), "</p>")
        summary_table(io, sets, on_device(sides, device), cand, base, wall)
    end
    println(io, "<p class=\"m\">Benchmark $BENCHMARK_VERSION fixes the instances, the options, Ipopt and HSL $HSL_VERSION.</p>")
    base === nothing || println(io, "<p class=\"m\">Base: ", escape_html(base.label), "</p>")
    println(io, "<p class=\"m\">SGM: shifted geometric mean of the solve time, with a shift of 10 s and a failure counting as ", round(Int, wall), " s, over the problems that all solvers shown have run. SSGM: the SGM scaled so that Ipopt, or the best solver where Ipopt has not run, is 1.</p>")
    page(joinpath(site, "index.html"), title, String(take!(io)), bar; refresh = !finished)
    return
end

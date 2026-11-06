# The benchmark report: a static site with a front page of the machines, the summary table and
# Dolan-Moré performance profiles, a page per set with every instance, and the result files
# and solver logs under `data/`.

const COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

const STYLE = """
body{font:14px/1.45 system-ui,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:110em;margin:1em auto;padding:0 1em}
.wide{overflow-x:auto;-webkit-overflow-scrolling:touch}
svg{display:block;width:100%;max-width:620px;height:auto}
@media(max-width:640px){body{font-size:13px;padding:0 .6em}td,th{padding:3px 6px}.prof{display:block;margin-right:0}input{width:100%;box-sizing:border-box}}
h1{font-size:1.5em}h2{font-size:1.15em;margin-top:2em}a{color:#1c5fb0}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid #e4e3df;padding:3px 10px;text-align:right;white-space:nowrap}
td:first-child,th:first-child,td.l,th.l{text-align:left}
th{color:#52514e;font-weight:600;cursor:default}
th.s{cursor:pointer}pre{background:#f3f2ee;padding:8px;overflow:auto}
.m{color:#52514e}.fail{font-weight:600}svg text{font:12px system-ui,sans-serif;fill:#52514e}
.key{display:inline-block;margin-right:1.2em}.key i{display:inline-block;width:14px;height:3px;border-radius:2px;margin:0 6px 3px 0}
.prof{position:relative;display:inline-block;vertical-align:top;margin:0 1.5em 1em 0}
.tip{position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #d6d5cf;border-radius:4px;padding:4px 8px;font-size:12px;white-space:nowrap}
input{font:inherit;padding:3px 6px;margin-bottom:8px;width:20em}
"""

const SCRIPT = """
document.querySelectorAll('.prof').forEach(p=>{const s=p.querySelector('svg'),t=p.querySelector('.tip'),l=p.querySelector('.x');
const d=JSON.parse(s.dataset.c);
s.addEventListener('mousemove',e=>{const r=s.getBoundingClientRect(),x=(e.clientX-r.left)*d.w/r.width;
if(x<d.l||x>d.r){t.style.display='none';l.style.display='none';return}
const tau=Math.pow(2,(x-d.l)/(d.r-d.l)*d.x);
t.innerHTML='within '+tau.toFixed(2)+'× of the best<br>'+d.s.map(c=>c[0]+': '+(100*c[1].filter(v=>v<=tau).length/d.n).toFixed(1)+'%').join('<br>');
t.style.display='block';t.style.left=Math.max(0,Math.min(e.clientX-r.left+12,r.width-200))+'px';t.style.top='8px';
l.setAttribute('x1',x);l.setAttribute('x2',x);l.style.display='block'});
s.addEventListener('mouseleave',()=>{t.style.display='none';l.style.display='none'})});
const tb=document.querySelector('table.cases');
if(tb){const body=tb.tBodies[0];
document.querySelector('#filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
for(const r of body.rows)r.style.display=r.cells[0].textContent.toLowerCase().includes(q)?'':'none'});
tb.querySelectorAll('th.s').forEach(h=>{let up=true;h.addEventListener('click',()=>{const i=h.cellIndex;up=!up;
const key=r=>{const v=r.cells[i].textContent,n=parseFloat(v);return isNaN(n)?v:n};
[...body.rows].sort((a,b)=>{const x=key(a),y=key(b),c=x<y?-1:x>y?1:0;return up?-c:c}).forEach(r=>body.append(r))})})}
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

# The columns of a report: one per solver and configuration of a side.
struct Side
    label::String
    color::String
    key::Tuple{String, String}   # device, configuration
    base::Bool
end

function report_sides(cand, base)
    label(key) = key[2] == "ipopt" ? "Ipopt" : solver_name(key[1]) * (key[2] == "default" ? "" : " " * key[2])
    keys_of(rs) = rs === nothing ? [] : sort!(unique!([(k[1], k[3]) for k in keys(rs.rows)]); by = k -> (k[1] != "cpu", k))
    sides = [(label(k), k, false) for k in keys_of(cand)]
    append!(sides, [(label(k) * " (base)", k, true) for k in keys_of(base)])
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

# One profile per device: a solver on the host and one on a GPU are never compared.
function profile_section(io, set, sides, cand, base, wall)
    for device in unique!([s.key[1] for s in sides])
        device_profile(io, set, device, [s for s in sides if s.key[1] == device], cand, base, wall)
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

function page(path, title, content; refresh = false)
    tmp = path * ".tmp"
    write(
        tmp, "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        refresh ? "<meta http-equiv=\"refresh\" content=\"30\">" : "",
        "<title>", escape_html(title), "</title><style>", STYLE, "</style><body>", content, "<script>", SCRIPT, "</script></body></html>\n",
    )
    chmod(tmp, 0o644)
    mv(tmp, path; force = true)
    return
end

function summary_table(io, sets, sides, cand, base, wall)
    println(io, "<div class=\"wide\"><table><thead><tr><th>set</th><th class=\"l\">solver</th><th>instances</th><th>solved</th><th>SGM (s)</th><th>small</th><th>medium</th><th>large</th>", base === nothing ? "" : "<th>time vs base</th>", "</tr></thead><tbody>")
    for set in sets, s in sides
        rows = side_rows(s, cand, base, set)
        isempty(rows) && continue
        all_rows = collect(values(rows))
        cells = map((:small, :medium, :large)) do class
            sel = [r for r in all_rows if size_class(r) == class]
            isempty(sel) ? "–" : @sprintf("%d / %d, %.2f", count(solved, sel), length(sel), sgm(score_time.(sel, wall)))
        end
        print(io, "<tr><td><a href=\"$set.html\">$set</a></td><td class=\"l\">$(escape_html(s.label))</td><td>$(length(all_rows))</td><td>$(count(solved, all_rows))</td>")
        @printf(io, "<td>%.2f</td>", sgm(score_time.(all_rows, wall)))
        foreach(c -> print(io, "<td>", c, "</td>"), cells)
        if base !== nothing
            other = s.base ? nothing : get(base.rows, (s.key[1], set, s.key[2]), nothing)
            common = other === nothing ? String[] : collect(intersect(keys(rows), keys(other)))
            if isempty(common)
                print(io, "<td>–</td>")
            else
                ga = sgm([score_time(other[c], wall) for c in common]); gb = sgm([score_time(rows[c], wall) for c in common])
                @printf(io, "<td>%.2f</td>", (gb + 10) / (ga + 10))
            end
        end
        println(io, "</tr>")
    end
    println(io, "</tbody></table></div>")
    return
end

function set_page(site, set, sides, cand, base, wall, logs, heading; refresh)
    rows = [side_rows(s, cand, base, set) for s in sides]
    keep = [i for i in eachindex(sides) if !isempty(rows[i])]
    sides, rows = sides[keep], rows[keep]
    io = IOBuffer()
    println(io, "<p><a href=\"index.html\">Benchmark report</a></p><h1>", escape_html(set), "</h1><p class=\"m\">", heading, "</p>")
    println(io, "<h2>Performance profile</h2>")
    profile_section(io, set, sides, cand, base, wall)
    println(io, "<h2>Instances</h2><input id=\"filter\" placeholder=\"Filter by name\">")
    order = sortperm([(s.key[1] != "cpu", s.base) for s in sides])
    sides, rows = sides[order], rows[order]
    println(io, "<div class=\"wide\"><table class=\"cases\"><thead><tr><th class=\"s\" rowspan=\"2\">instance</th><th class=\"s\" rowspan=\"2\">variables</th><th class=\"s\" rowspan=\"2\">constraints</th>")
    foreach(s -> print(io, "<th colspan=\"6\" class=\"l\">", escape_html(s.label), "</th>"), sides)
    print(io, "</tr><tr>")
    foreach(_ -> print(io, "<th class=\"s l\">status</th><th class=\"s\">iterations</th><th class=\"s\">time (s)</th><th class=\"s\">objective</th><th class=\"s\">infeasibility</th><th>log</th>"), sides)
    println(io, "</tr></thead><tbody>")
    for case in sort!(collect(reduce(union, [Set(keys(r)) for r in rows]; init = Set{String}())))
        any_row = first(r[case] for r in rows if haskey(r, case))
        print(io, "<tr><td>", escape_html(case), "</td><td>", any_row.nvar, "</td><td>", any_row.ncon, "</td>")
        for (s, r) in zip(sides, rows)
            if !haskey(r, case)
                print(io, "<td class=\"l\">–</td><td></td><td></td><td></td><td></td><td></td>")
                continue
            end
            row = r[case]
            link = s.base ? nothing : get(logs, (s.key[1], set, case, s.key[2]), nothing)
            print(io, "<td class=\"l", solved(row) ? "" : " fail", "\">", status_name(row.status), "</td><td>", row.iters, "</td>")
            @printf(io, "<td>%.3f</td><td>%.6g</td><td>%.1e</td>", row.t_total, row.obj, row.primal_feas)
            print(io, "<td>", link === nothing ? "" : "<a href=\"$link\">log</a>", "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</tbody></table></div>")
    page(joinpath(site, "$set.html"), "$set: MadNLP benchmark", String(take!(io)); refresh)
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
    write_report(site, root; base = nothing, wall = 7200.0, title, finished = false)

Write the report of the result directories under `root` to the directory `site`. `base` is a
`ResultSet` to compare with. While a run is `finished = false` the pages reload themselves and
the front page shows what every worker is on.
"""
function write_report(site, root; base = nothing, wall = 7200.0, title = "MadNLP benchmark", finished::Bool = false)
    mkpath(site)
    chmod(site, 0o755)
    logs = sync_data(site, root)
    cand = from_directory(root, "this run")
    sides = report_sides(cand, base)
    sets = sort!(unique!([k[2] for k in keys(cand.rows)]))
    stamp = Dates.format(now(UTC), "yyyy-mm-dd HH:MM:SS")
    heading = (finished ? "Finished" : "In progress") * ", updated $stamp UTC."
    io = IOBuffer()
    println(io, "<h1>", escape_html(title), "</h1><p class=\"m\">", heading, "</p>")
    println(io, "<h2>Machines</h2><ul>")
    for device in sort!(collect(keys(cand.machines)))
        println(io, "<li>", solver_name(device), ": ", escape_html(join(cand.machines[device], "; ")), "</li>")
    end
    if base !== nothing
        println(io, "<li>Base, ", escape_html(base.label), ": ", escape_html(join(unique!(reduce(vcat, collect(values(base.machines)))), "; ")), "</li>")
    end
    println(io, "</ul>")
    if !finished
        println(io, "<h2>Workers</h2>")
        for dir in sort!(readdir(root; join = true))
            file = joinpath(dir, "summary.md")
            isfile(file) || continue
            m = match(r"^```\n(.*?)\n```"s, read(file, String))
            m === nothing || println(io, "<p>", basename(dir), "</p><pre>", escape_html(m[1]), "</pre>")
        end
    end
    println(io, "<h2>Summary</h2><p class=\"m\">SGM is the shifted geometric mean of the solve time (shift 10 s, a failure counts as ", round(Int, wall), " s). Size classes give solved / instances and SGM.</p>")
    for device in unique!([s.key[1] for s in sides])
        println(io, "<h3>", device == "cuda" ? "GPU" : "CPU", "</h3>")
        summary_table(io, sets, [s for s in sides if s.key[1] == device], cand, base, wall)
    end
    for set in sets
        println(io, "<h2><a href=\"$set.html\">", set, "</a></h2>")
        kept = [s for s in sides if !isempty(side_rows(s, cand, base, set))]
        profile_section(io, set, kept, cand, base, wall)
        set_page(site, set, sides, cand, base, wall, logs, heading; refresh = !finished)
    end
    println(io, "<h2>Files</h2><p><a href=\"data/\">Result files and solver logs</a></p>")
    page(joinpath(site, "index.html"), title, String(take!(io)); refresh = !finished)
    return
end

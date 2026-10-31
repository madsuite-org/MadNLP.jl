"""
    EventTimer{E} <: MadNLP.AbstractTimer

Times regions on a GPU stream with device events of type `E`, without
synchronizing the host. [`MadNLP.timer_start!`](@ref) and
[`MadNLP.timer_stop!`](@ref) record an event on the current stream. A region's
duration is added to the counters once its stop event has completed: completed
regions are collected at every `timer_stop!` with a non-blocking query, and the
rest in [`MadNLP.timer_flush!`](@ref) at the end of the solve.

The measured time is the span of the device timeline between the two events, so
it includes the device work of the region and any time the device sits idle
while the host is still inside the region.

A backend provides `_new_event(::Type{E})`, `_record!(::E)`, `_isdone(::E)`,
`_wait(::E)` and `_elapsed(start::E, stop::E)` (in seconds).
"""
struct EventTimer{E} <: MadNLP.AbstractTimer
    starts::Vector{E}
    pending::Vector{Tuple{E, E, Symbol}}
    pool::Vector{E}
end
EventTimer{E}() where {E} = EventTimer{E}(E[], Tuple{E, E, Symbol}[], E[])

function _new_event end
function _record! end
function _isdone end
function _wait end
function _elapsed end

function _take_event!(timer::EventTimer{E}) where {E}
    return isempty(timer.pool) ? _new_event(E) : pop!(timer.pool)
end

function MadNLP.timer_start!(timer::EventTimer)
    start = _take_event!(timer)
    _record!(start)
    push!(timer.starts, start)
    return
end

function MadNLP.timer_stop!(timer::EventTimer, cnt, ::Val{field}) where {field}
    stop = _take_event!(timer)
    _record!(stop)
    push!(timer.pending, (pop!(timer.starts), stop, field))
    _collect!(timer, cnt, false)
    return
end

MadNLP.timer_flush!(timer::EventTimer, cnt) = _collect!(timer, cnt, true)

# Events on one stream complete in order, so collection stops at the first
# region that has not completed.
function _collect!(timer::EventTimer, cnt, blocking::Bool)
    while !isempty(timer.pending)
        start, stop, field = first(timer.pending)
        if !_isdone(stop)
            blocking || break
            _wait(stop)
        end
        setfield!(cnt, field, getfield(cnt, field) + _elapsed(start, stop))
        popfirst!(timer.pending)
        push!(timer.pool, start, stop)
    end
    return
end

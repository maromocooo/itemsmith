import { useEffect, useMemo, useRef, useState } from "react";
import { DetectedObject, Session, cropUrl, imageUrl } from "./api";

/**
 * Original photo with clickable boxes. Small boxes are drawn last so that an object
 * lying inside a larger one (mouse on a mouse pad) stays clickable.
 */
export function PhotoPane({
  session,
  objects,
  activeId,
  selectedIds,
  onHover,
  onSelect,
}: {
  session: Session;
  objects: DetectedObject[];
  activeId: string | null;
  selectedIds?: Set<string>;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const { width, height } = session.image;
  const ordered = useMemo(
    () => [...objects].sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3]),
    [objects],
  );
  const hovered = objects.find((o) => o.id === activeId);
  const active = hovered ? objects.filter((o) => o === hovered || (hovered.group && o.group === hovered.group)) : [];
  const unit = Math.max(width, height) / 1000;

  return (
    <div className="photo-frame">
      <img src={imageUrl(session.id)} alt={session.image.filename} width={width} height={height} />
      <svg viewBox={`0 0 ${width} ${height}`} onMouseLeave={() => onHover(null)} role="group" aria-label="検出されたオブジェクト">
        {ordered.map((o) => {
          const [x, y, w, h] = o.bbox;
          return (
            <g
              key={o.id}
              className={`box ${o.status} ${o.excluded ? "excluded" : ""} ${active.includes(o) ? "active" : ""} ${selectedIds?.has(o.id) ? "selected" : ""}`}
              onMouseEnter={() => onHover(o.id)}
              onClick={() => onSelect(o.id)}
              role="button"
              aria-label={`${o.number} ${o.label}`}
            >
              <rect x={x} y={y} width={w} height={h} vectorEffect="non-scaling-stroke" />
            </g>
          );
        })}
        {active.length > 0 && (
          <g className="spotlight" pointerEvents="none">
            <path
              fillRule="evenodd"
              d={`M0 0H${width}V${height}H0Z ${active.map(({ bbox: [x, y, w, h] }) => `M${x} ${y}h${w}v${h}h${-w}Z`).join(" ")}`}
            />
            {active.map((o) => (
              <g key={o.id}>
                <rect x={o.bbox[0]} y={o.bbox[1]} width={o.bbox[2]} height={o.bbox[3]} vectorEffect="non-scaling-stroke" />
                <BoxTag obj={o} unit={unit} imageWidth={width} />
              </g>
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}

function BoxTag({ obj, unit, imageWidth }: { obj: DetectedObject; unit: number; imageWidth: number }) {
  const text = `${obj.number}. ${obj.label}`;
  const size = 16 * unit;
  const tagWidth = Math.min(Array.from(text).length * size * 0.95 + size, imageWidth);
  const tagHeight = size * 1.7;
  const x = Math.max(0, Math.min(obj.bbox[0], imageWidth - tagWidth));
  const y = obj.bbox[1] > tagHeight ? obj.bbox[1] - tagHeight : obj.bbox[1];
  return (
    <>
      <rect className="tag-bg" x={x} y={y} width={tagWidth} height={tagHeight} rx={3 * unit} />
      <text x={x + size / 2} y={y + size * 1.2} fontSize={size}>{text}</text>
    </>
  );
}

export function ProductImages({ session, members }: { session: Session; members: DetectedObject[] }) {
  return (
    <div className={`crops ${members.length > 2 ? "many" : ""}`}>
      {members.map((o) => <img key={o.id} className="crop" src={cropUrl(session.id, o.id)} alt="" loading="lazy" />)}
    </div>
  );
}

export function ProductTitle({ members }: { members: DetectedObject[] }) {
  return (
    <>
      {members.map((o) => <span key={o.id} className="number">{o.number}</span>)}
      <span className="label">{members.map((o) => o.label).join(" / ")}</span>
    </>
  );
}

/** Shared row <-> photo linking: hover highlights both, a box click jumps to its row. */
export function useLinkedRow(focusSelector: string) {
  const rows = useRef(new Map<string, HTMLElement>());
  const timer = useRef<number | undefined>(undefined);
  const [flashId, setFlashId] = useState<string | null>(null);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const register = (ids: string[]) => (el: HTMLElement | null) => {
    for (const id of ids) {
      if (el) rows.current.set(id, el);
      else rows.current.delete(id);
    }
  };

  const jumpTo = (id: string) => {
    const row = rows.current.get(id);
    if (!row) return;
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashId(id);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setFlashId(null), 1600);
    row.querySelector<HTMLElement>(focusSelector)?.focus({ preventScroll: true });
  };

  return { register, jumpTo, flashId };
}

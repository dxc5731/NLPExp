export type Entity = { text: string; entity_type: string; start: number; end: number; tags: string[] };
export function segments(text: string, entities: Entity[]) {
  const chars = Array.from(text); let pos = 0;
  const result: { text: string; entity?: Entity }[] = [];
  for (const entity of [...entities].sort((a, b) => a.start - b.start)) {
    if (entity.start < pos || entity.end > chars.length || entity.end <= entity.start) throw new Error('实体字符区间无效');
    if (entity.start > pos) result.push({ text: chars.slice(pos, entity.start).join('') });
    const piece = chars.slice(entity.start, entity.end).join('');
    if (piece !== entity.text) throw new Error('实体区间与原文不一致');
    result.push({ text: piece, entity }); pos = entity.end;
  }
  if (pos < chars.length) result.push({ text: chars.slice(pos).join('') });
  return result;
}
export const percent = (v: number | null | undefined) => v == null ? '—' : `${(v * 100).toFixed(4)}%`;
export const delta = (v: number, baseline: number) => `${v >= baseline ? '+' : ''}${((v - baseline) * 100).toFixed(4)} pp`;

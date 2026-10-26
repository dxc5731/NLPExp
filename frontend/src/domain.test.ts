import { describe, it, expect } from 'vitest';
import { segments, delta, percent } from './domain';
describe('原始Unicode坐标与结果显示', () => {
  it('按码点恢复含空格、英文和补充平面字符的原文', () => {
    const text = 'A 😀发热。'; const e = { text: '发热', entity_type: 'Symptom', start: 3, end: 5, tags: ['B-Symptom', 'I-Symptom'] };
    expect(segments(text, [e]).map(x => x.text).join('')).toBe(text);
    expect(segments(text, [e])[1].text).toBe('发热');
  });
  it('无实体保留全部原文', () => expect(segments(' 今天好 ', []).map(x=>x.text).join('')).toBe(' 今天好 '));
  it('拒绝错误的UTF16跨度', () => expect(()=>segments('😀发热', [{text:'发热',entity_type:'Symptom',start:2,end:4,tags:[]}])).toThrow());
  it('无本次评分显示横线', () => expect(percent(null)).toBe('—'));
  it('差值显示百分点', () => expect(delta(0.9009,0.8999)).toBe('+0.1000 pp'));
});

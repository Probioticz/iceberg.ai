import React from 'react';
export default function AgeSelector({ ageGroup, setAgeGroup }) {
  return (
    <div>
      <label>Choose your age group: </label>
      <select value={ageGroup} onChange={e => setAgeGroup(Number(e.target.value))}>
        <option value={9}>9+</option>
        <option value={13}>13+</option>
        <option value={16}>16+</option>
      </select>
    </div>
  );
}

const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, VerticalAlign,
} = require("docx");

const PAGE_WIDTH_DXA = 12240; // US Letter
const MARGIN = 1080; // 0.75in
const USABLE_WIDTH = PAGE_WIDTH_DXA - MARGIN * 2;

function hr() {
  return new Paragraph({
    border: { bottom: { color: "999999", space: 1, style: BorderStyle.SINGLE, size: 6 } },
    spacing: { after: 200 },
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 120 }, children: [new TextRun({ text, bold: true })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 }, children: [new TextRun({ text, bold: true })] });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 140 }, children: [new TextRun({ text, ...opts })] });
}
function bullet(text, opts = {}) {
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text, ...opts })] });
}

function cell(text, { width, bold = false, shade = null, align = AlignmentType.LEFT, size = 20 } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, bold, size })] })],
  });
}

const colWidths = [1900, 2600, 2200, 2200, 1600];
const tableWidth = colWidths.reduce((a, b) => a + b, 0);

// Header cell builder with white text on dark shading
function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "2F5496" },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 20 })] })],
  });
}

const table = new Table({
  width: { size: tableWidth, type: WidthType.DXA },
  columnWidths: colWidths,
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        headerCell("Joint", colWidths[0]),
        headerCell("Angle Range Required", colWidths[1]),
        headerCell("Minimum Torque (performance floor)", colWidths[2]),
        headerCell("Simulated Torque Spec (validated in training)", colWidths[3]),
        headerCell("Your Planned Scheme", colWidths[4]),
      ],
    }),
    new TableRow({
      children: [
        cell("Knee\n(knee_hinge)", { width: colWidths[0], bold: true }),
        cell("–60° to +90°\n(150° span)", { width: colWidths[1] }),
        cell("~5 N·m combined\n(found sufficient to pump well; not tested down to a hard failure point)", { width: colWidths[2] }),
        cell("5.4 N·m combined\n(2× Feetech STS3215-C044, ~2.69 N·m stall each)", { width: colWidths[3] }),
        cell("2 servos — MATCHES", { width: colWidths[4], shade: "D9EAD3" }),
      ],
    }),
    new TableRow({
      children: [
        cell("Lean / Torso\n(lean_hinge)", { width: colWidths[0], bold: true }),
        cell("–5° to +80°\n(85° span)", { width: colWidths[1] }),
        cell("~8 N·m combined\n(hard threshold — below this, joint could not pump at all in simulation)", { width: colWidths[2] }),
        cell("8.1 N·m combined\n(3× Feetech STS3215-C044, ~2.69 N·m stall each)", { width: colWidths[3] }),
        cell("1 servo — DOES NOT MATCH, see below", { width: colWidths[4], shade: "F4CCCC" }),
      ],
    }),
  ],
});

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH_DXA, height: 15840 },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      children: [
        new Paragraph({
          heading: HeadingLevel.TITLE,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Swing Robot — Servo Shopping Requirements", bold: true, size: 32 })],
        }),
        p("Reference: swing_robot.xml actuator section, README.md §2/§5.9. Updated August 17, 2026.", { italics: true, size: 20 }),
        hr(),

        h1("1. Actuation Scheme"),
        p("Planned build: 2 synchronized servos on the knee joint, 1 servo on the lean/torso joint. This matches the knee configuration actually validated in simulation. It does NOT match the lean/torso configuration — see Section 3."),

        h1("2. Per-Joint Requirements"),
        table,
        new Paragraph({ spacing: { before: 160, after: 100 } }),
        p("\"Minimum torque\" is the performance floor this project actually tested for — not a guess. \"Simulated torque spec\" is the exact number the current trained policy (best_model) was validated against; this is the number to match or exceed, not the bare minimum, if you want the trained policy's real-world performance to resemble simulation.", { size: 20, italics: true }),

        h1("3. IMPORTANT — Lean/Torso Servo Shortfall"),
        p("A single Feetech STS3215-C044 (~2.69 N·m stall) provides roughly a third of the ~8 N·m the simulation found necessary just for the lean/torso joint to pump AT ALL — not the 8.1 N·m actually used to reach the trained policy's target performance. Going with only 1 servo on this joint, as currently planned, means one of the following has to happen:", { bold: true }),
        bullet("Choose a single servo rated at least ~8 N·m stall torque (ideally with headroom above 8.1 N·m, since stall ratings are a ceiling, not a sustained/continuous rating) for the lean/torso joint specifically — a materially larger, more expensive unit than the Feetech STS3215 class used elsewhere in this analysis."),
        bullet("Revisit the scheme to use a synchronized multi-servo setup on lean/torso as well, matching how knee was actually validated (e.g. 3× STS3215-C044 as simulated, or 2× of a higher-torque model)."),
        bullet("Accept reduced real-world performance vs. the simulated/trained policy — the trained policy has not been evaluated against an under-torqued lean joint, and the mass-sensitivity investigation (§6 below) found this exact joint's torque budget already sits close to a sharp performance cliff even at the full 8.1 N·m."),
        p("This needs a decision before ordering parts — it changes which servo model is appropriate for that joint.", { bold: true }),

        h1("4. Control Mode Requirements"),
        p("Both joints were simulated and trained as POSITION-controlled actuators (MuJoCo position/PD servo, kp=40, dampratio=1 — critically damped, stiff tracking). Whatever hardware is chosen needs to support closed-loop position control (a target-angle command, not open-loop speed or raw torque control), and needs to be able to command across the full angle range in Section 2 without hitting internal servo limits."),

        h1("5. Not Covered by This Simulation — Verify Separately"),
        p("Only torque (from a direct pumping-capability sweep) and position-control mode are grounded in simulation evidence. The following were never tested and are NOT part of this spec — don't assume simulation validates them:"),
        bullet("Response speed / bandwidth (how fast the servo can reach a commanded position)"),
        bullet("Backlash / gear-train play"),
        bullet("Duty cycle, thermal limits, or sustained-load rating (only stall/peak torque was used)"),
        bullet("Real-world friction and gearbox losses (this project's joint damping values are placeholders, sensitivity-tested but not measured — see README §5.9r/damping sweep)"),

        h1("6. Related: Torso/Head Mass Budget Shares the Same Torque Budget"),
        p("A prior mass-sensitivity investigation (README §5.9r) found torso mass up to 3.0 kg and head mass up to 0.6 kg are safe under the CURRENT 8.1 N·m lean/torso torque budget, with a sharp performance cliff just above each (torso: fails by 3.5 kg; head: fails by 0.7 kg). That cliff is directly tied to how much of the lean servo's torque is left over after moving the rider's own mass — a WEAKER lean/torso servo (per Section 3's shortfall) would very plausibly pull those safe mass limits down further, and the servo's own added mass at the joint counts against the same budget too. Worth re-running that sensitivity check once a final lean/torso servo is chosen, not just assuming the existing 3.0 kg / 0.6 kg figures still hold."),

        hr(),
        p("Source data: swing_robot.xml <actuator> section (forcerange, ctrlrange), README.md §2 (mechanism description) and §5.9r (mass-sensitivity sweep). This document reflects the simulation state as of the domain-randomization validation completed August 17, 2026.", { size: 18, italics: true, color: "666666" }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync("servo_shopping_requirements.docx", buf);
  console.log("wrote servo_shopping_requirements.docx");
});

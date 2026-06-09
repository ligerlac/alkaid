`timescale 1ns / 1ps


module ternary_adder #(
    parameter BW_INPUT0 = 32,
    parameter SIGNED0 = 0,
    parameter NEGATE0 = 0,
    parameter PAD0 = 0,
    parameter BW_INPUT1 = 32,
    parameter SIGNED1 = 0,
    parameter NEGATE1 = 0,
    parameter PAD1 = 0,
    parameter BW_INPUT2 = 32,
    parameter SIGNED2 = 0,
    parameter NEGATE2 = 0,
    parameter PAD2 = 0,
    parameter BW_OUT = 32,
    parameter DROP_LSBS = 0
) (
    input [BW_INPUT0-1:0] in0,
    input [BW_INPUT1-1:0] in1,
    input [BW_INPUT2-1:0] in2,
    output [BW_OUT-1:0] out
);

  function integer max2;
    input integer lhs;
    input integer rhs;
    begin
      max2 = (lhs > rhs) ? lhs : rhs;
    end
  endfunction

  function integer min2;
    input integer lhs;
    input integer rhs;
    begin
      min2 = (lhs < rhs) ? lhs : rhs;
    end
  endfunction

  function integer select3;
    input integer sel;
    input integer val0;
    input integer val1;
    input integer val2;
    begin
      select3 = (sel == 0) ? val0 : (sel == 1) ? val1 : val2;
    end
  endfunction

  localparam OUT_HIGH_ABS = DROP_LSBS + BW_OUT - 1;
  localparam MIN_PAD = min2(min2(PAD0, PAD1), PAD2);
  localparam LOW_COUNT = ((PAD0 == MIN_PAD) ? 1 : 0) + ((PAD1 == MIN_PAD) ? 1 : 0) + ((PAD2 == MIN_PAD) ? 1 : 0);
  localparam LOW_INPUT = (PAD0 == MIN_PAD) ? 0 : (PAD1 == MIN_PAD) ? 1 : 2;
  localparam LOW_NEGATE = select3(LOW_INPUT, NEGATE0, NEGATE1, NEGATE2);
  localparam NO_NEXT_PAD = OUT_HIGH_ABS + 2;
  localparam NEXT_PAD0 = (PAD0 > MIN_PAD) ? PAD0 : NO_NEXT_PAD;
  localparam NEXT_PAD1 = (PAD1 > MIN_PAD) ? PAD1 : NO_NEXT_PAD;
  localparam NEXT_PAD2 = (PAD2 > MIN_PAD) ? PAD2 : NO_NEXT_PAD;
  localparam UPPER_START = min2(min2(NEXT_PAD0, NEXT_PAD1), NEXT_PAD2);
  localparam LOW_COPY_ABS_LO = max2(DROP_LSBS, MIN_PAD);
  localparam LOW_COPY_ABS_HI = min2(OUT_HIGH_ABS, UPPER_START - 1);
  localparam LOWCOPY_ENABLE = (LOW_COUNT == 1) && (LOW_NEGATE == 0) && (UPPER_START != NO_NEXT_PAD) &&
                              (UPPER_START > DROP_LSBS) && (LOW_COPY_ABS_HI >= LOW_COPY_ABS_LO);

  localparam FULL_MAX_TERM_BW = max2(max2(BW_INPUT0 + PAD0, BW_INPUT1 + PAD1), BW_INPUT2 + PAD2);
  localparam FULL_BW_ADD = max2(FULL_MAX_TERM_BW + 4, BW_OUT + DROP_LSBS + 1);
  localparam FULL_LEFT_PAD0 = FULL_BW_ADD - BW_INPUT0 - PAD0;
  localparam FULL_LEFT_PAD1 = FULL_BW_ADD - BW_INPUT1 - PAD1;
  localparam FULL_LEFT_PAD2 = FULL_BW_ADD - BW_INPUT2 - PAD2;

  generate
    if (LOWCOPY_ENABLE) begin : lowcopy
      localparam LOW_PAD = select3(LOW_INPUT, PAD0, PAD1, PAD2);
      localparam LOW_OUT_LO = LOW_COPY_ABS_LO - DROP_LSBS;
      localparam LOW_OUT_HI = LOW_COPY_ABS_HI - DROP_LSBS;
      localparam ZERO_HI = min2(OUT_HIGH_ABS, LOW_PAD - 1) - DROP_LSBS;
      localparam SUFFIX_BW_RAW = OUT_HIGH_ABS - UPPER_START + 1;
      localparam SUFFIX_BW = (SUFFIX_BW_RAW > 0) ? SUFFIX_BW_RAW : 1;
      localparam SUFFIX_MAX_TERM_BW =
          max2(max2(max2(BW_INPUT0 + PAD0 - UPPER_START, 1), max2(BW_INPUT1 + PAD1 - UPPER_START, 1)),
               max2(BW_INPUT2 + PAD2 - UPPER_START, 1));
      localparam SUFFIX_BW_ADD = max2(SUFFIX_MAX_TERM_BW + 4, SUFFIX_BW + 1);
      localparam EXT_BW = max2(DROP_LSBS + BW_OUT, UPPER_START + SUFFIX_BW_ADD);
      localparam LEFT_PAD0 = EXT_BW - BW_INPUT0 - PAD0;
      localparam LEFT_PAD1 = EXT_BW - BW_INPUT1 - PAD1;
      localparam LEFT_PAD2 = EXT_BW - BW_INPUT2 - PAD2;

      // verilator lint_off UNUSEDSIGNAL
      wire [EXT_BW-1:0] in0_ext;
      wire [EXT_BW-1:0] in1_ext;
      wire [EXT_BW-1:0] in2_ext;

      if (SIGNED0 == 1) begin : in0_is_signed
        assign in0_ext = {{LEFT_PAD0{in0[BW_INPUT0-1]}}, in0, {PAD0{1'b0}}};
      end else begin : in0_is_unsigned
        assign in0_ext = {{LEFT_PAD0{1'b0}}, in0, {PAD0{1'b0}}};
      end

      if (SIGNED1 == 1) begin : in1_is_signed
        assign in1_ext = {{LEFT_PAD1{in1[BW_INPUT1-1]}}, in1, {PAD1{1'b0}}};
      end else begin : in1_is_unsigned
        assign in1_ext = {{LEFT_PAD1{1'b0}}, in1, {PAD1{1'b0}}};
      end

      if (SIGNED2 == 1) begin : in2_is_signed
        assign in2_ext = {{LEFT_PAD2{in2[BW_INPUT2-1]}}, in2, {PAD2{1'b0}}};
      end else begin : in2_is_unsigned
        assign in2_ext = {{LEFT_PAD2{1'b0}}, in2, {PAD2{1'b0}}};
      end

      wire signed [SUFFIX_BW_ADD-1:0] term0 =
          (NEGATE0 == 1) ? -$signed(in0_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START])
                         :  $signed(in0_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START]);
      wire signed [SUFFIX_BW_ADD-1:0] term1 =
          (NEGATE1 == 1) ? -$signed(in1_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START])
                         :  $signed(in1_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START]);
      wire signed [SUFFIX_BW_ADD-1:0] term2 =
          (NEGATE2 == 1) ? -$signed(in2_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START])
                         :  $signed(in2_ext[UPPER_START+SUFFIX_BW_ADD-1:UPPER_START]);
      wire signed [SUFFIX_BW_ADD-1:0] accum = term0 + term1 + term2;
      // verilator lint_on UNUSEDSIGNAL

      if (LOW_PAD > DROP_LSBS) begin : zero_low_bits
        assign out[ZERO_HI:0] = {(ZERO_HI + 1){1'b0}};
      end

      if (LOW_INPUT == 0) begin : low_from_in0
        assign out[LOW_OUT_HI:LOW_OUT_LO] = in0_ext[LOW_COPY_ABS_HI:LOW_COPY_ABS_LO];
      end else if (LOW_INPUT == 1) begin : low_from_in1
        assign out[LOW_OUT_HI:LOW_OUT_LO] = in1_ext[LOW_COPY_ABS_HI:LOW_COPY_ABS_LO];
      end else begin : low_from_in2
        assign out[LOW_OUT_HI:LOW_OUT_LO] = in2_ext[LOW_COPY_ABS_HI:LOW_COPY_ABS_LO];
      end

      if (UPPER_START <= OUT_HIGH_ABS) begin : add_upper_bits
        assign out[BW_OUT-1:UPPER_START-DROP_LSBS] = accum[SUFFIX_BW-1:0];
      end
    end else begin : full
      // verilator lint_off UNUSEDSIGNAL
      wire [FULL_BW_ADD-1:0] in0_ext;
      wire [FULL_BW_ADD-1:0] in1_ext;
      wire [FULL_BW_ADD-1:0] in2_ext;

      if (SIGNED0 == 1) begin : in0_is_signed
        assign in0_ext = {{FULL_LEFT_PAD0{in0[BW_INPUT0-1]}}, in0, {PAD0{1'b0}}};
      end else begin : in0_is_unsigned
        assign in0_ext = {{FULL_LEFT_PAD0{1'b0}}, in0, {PAD0{1'b0}}};
      end

      if (SIGNED1 == 1) begin : in1_is_signed
        assign in1_ext = {{FULL_LEFT_PAD1{in1[BW_INPUT1-1]}}, in1, {PAD1{1'b0}}};
      end else begin : in1_is_unsigned
        assign in1_ext = {{FULL_LEFT_PAD1{1'b0}}, in1, {PAD1{1'b0}}};
      end

      if (SIGNED2 == 1) begin : in2_is_signed
        assign in2_ext = {{FULL_LEFT_PAD2{in2[BW_INPUT2-1]}}, in2, {PAD2{1'b0}}};
      end else begin : in2_is_unsigned
        assign in2_ext = {{FULL_LEFT_PAD2{1'b0}}, in2, {PAD2{1'b0}}};
      end

      wire signed [FULL_BW_ADD-1:0] term0 = (NEGATE0 == 1) ? -$signed(in0_ext) : $signed(in0_ext);
      wire signed [FULL_BW_ADD-1:0] term1 = (NEGATE1 == 1) ? -$signed(in1_ext) : $signed(in1_ext);
      wire signed [FULL_BW_ADD-1:0] term2 = (NEGATE2 == 1) ? -$signed(in2_ext) : $signed(in2_ext);
      wire signed [FULL_BW_ADD-1:0] accum = term0 + term1 + term2;
      // verilator lint_on UNUSEDSIGNAL

      assign out = accum[BW_OUT-1+DROP_LSBS:DROP_LSBS];
    end
  endgenerate

endmodule

import math
import tkinter
from tkinter import ttk


def kyuyo_deduction_cal(salary):
    if salary < 650000:
        deduction = 650000
    elif 650000 <= salary <= 1800000:
        deduction = salary * 0.4
    elif 1800000 < salary <= 3600000:
        deduction = salary * 0.3 + 180000
    elif 3600000 < salary <= 6600000:
        deduction = salary * 0.2 + 540000
    elif 6600000 < salary <= 10000000:
        deduction = salary * 0.1 + 1200000
    else:
        deduction = 2200000
    return deduction

def old_cal(old):
    if old <= 25000:
        fee_old = old
    elif 25000 < old <= 50000:
        fee_old = old/2 + 12500
    elif 50000 < old <= 100000:
        fee_old = old/4 + 25000
    else:
        fee_old = 50000
    return fee_old
    
def new_cal(new):
    if new <= 20000:
        fee_new = new
    elif 20000 < new <= 40000:
        fee_new = new/2 + 10000
    elif 40000 < new <= 80000:
        fee_new = new/4 + 20000
    else:
        fee_new = 40000
    return fee_new

def seiho_deduction_cal(old_seiho, new_seiho, kaigo):
    fee_seiho_old = old_cal(old_seiho)
    fee_seiho_new = new_cal(new_seiho)
    fee_kaigo = new_cal(kaigo)
    if fee_seiho_old > 40000:
        fee_sum = fee_seiho_old + fee_kaigo
    else:
        fee_sum = min([(fee_seiho_new + fee_seiho_old), 40000]) + fee_kaigo
    if new_seiho == 0 and kaigo == 0:
        fee_sum = min([fee_sum, 100000])
    else:
        fee_sum = min([fee_sum, 120000])
    return fee_sum

def income_tax_cal(income):
    if income <= 1950000:
        tax = income * 0.05
        tax_rate = 0.05
    elif 1950000 < income <= 3300000:
        tax = income * 0.1 - 97500
        tax_rate = 0.1
    elif 3300000 < income <= 6950000:
        tax = income * 0.2 - 427500
        tax_rate = 0.2
    elif 6950000 < income <= 9000000:
        tax = income * 0.23 - 636000
        tax_rate = 0.23
    elif 9000000 < income <= 18000000:
        tax = income * 0.33 - 1536000
        tax_rate = 0.33
    elif 18000000 < income <= 40000000:
        tax = income * 0.4 - 2796000
        tax_rate = 0.4
    else:
        tax = income * 0.45 - 4796000
        tax_rate = 0.45
    return tax, tax_rate

def old2_cal(old):
    if old <= 15000:
        fee_old = old
    elif 15000 < old <= 40000:
        fee_old = old/2 + 7500
    elif 40000 < old <= 70000:
        fee_old = old/4 + 17500
    else:
        fee_old = 35000
    return fee_old

def new2_cal(new):
    if new <= 12000:
        fee_new = new
    elif 12000 < new <= 32000:
        fee_new = new/2 + 6000
    elif 32000 < new <= 56000:
        fee_new = new/4 + 14000
    else:
        fee_new = 28000
    return fee_new

def seiho_deduction2_cal(old_seiho, new_seiho, kaigo):
    fee_seiho_old = old2_cal(old_seiho)
    fee_seiho_new = new2_cal(new_seiho)
    fee_kaigo = new2_cal(kaigo)
    if fee_seiho_old > 28000:
        fee_sum = fee_seiho_old + fee_kaigo
    else:
        fee_sum = min([(fee_seiho_new + fee_seiho_old), 28000]) + fee_kaigo
    fee_sum = min([fee_sum, 70000])
    return fee_sum

def eq_deduction2_cal(eq):
    if eq <= 50000:
        fee = eq / 2
    else:
        fee = 25000
    return fee

def difference_of_human_deduction(syo_kiso, syo_fu, ju_kiso, ju_fu, kazeisyotoku):
    diff_human_deduction = syo_kiso + syo_fu - (ju_kiso + ju_fu)
    if kazeisyotoku <= 2000000:
        reduction = min([diff_human_deduction, kazeisyotoku]) * 0.05
    else:
        reduction =  max([(diff_human_deduction - (kazeisyotoku - 2000000)) * 0.05, 2500])
    return reduction
    

def tax_calc():
    in1 = float(in1_box.get())
    in2 = float(in2_box.get())
    in3 = float(in3_box.get())
    in4 = float(in4_box.get())
    in5 = float(in5_box.get())
    in6 = float(in6_box.get())
    in7 = float(in7_box.get())
    in8 = float(in8_box.get())
    in9 = float(in9_box.get())
    in10 = float(in10_box.get())
    in11 = float(in11_box.get())
    out1 = in1 - kyuyo_deduction_cal(in1)
    out1_label.configure(text='給与所得:' + str(out1))
    out2 = seiho_deduction_cal(in4, in3, in5)
    out2_label.configure(text='生命保険料控除:' + str(out2))
    out16 = min([in11, 50000])
    out16_label.configure(text='地震保険料控除:' + str(out16))
    out3 = in2 + out2 + in6 + in7 + out16
    out3_label.configure(text='所得控除の合計:' + str(out3))
    out4 = math.floor((out1 - out3)/1000) * 1000
    out4_label.configure(text='課税所得:' + str(out4))
    out5, tax_rate = income_tax_cal(out4)
    out5_label.configure(text='所得税:' + str(out5))
    out6 = seiho_deduction2_cal(in4, in3, in5)
    out6_label.configure(text='生命保険料控除:' + str(out6))
    out17 = eq_deduction2_cal(in11)
    out17_label.configure(text='地震保険料控除:' + str(out17))
    out7 = in2 + out6 + in8 + in9 + out17
    out7_label.configure(text='所得控除の合計:' + str(out7))
    out8 = math.floor((out1 - out7)/1000) * 1000
    out8_label.configure(text='課税所得:' + str(out8))
    out9 = out8 * 0.06
    out9_label.configure(text='市民税額:' + str(out9))
    out10 = out8 * 0.04
    out10_label.configure(text='県民税額:' + str(out10))
    out11 = 3500
    out11_label.configure(text='市均等割:' + str(out11))
    out12 = 2000
    out12_label.configure(text='県均等割:' + str(out12))
    reduction = difference_of_human_deduction(in6, in7, in8, in9, out8)
    out13 = (in10 - 2000) * tax_rate * 0.6 + (in10 - 2000) * 0.06 + (in10 - 2000) * (0.9 - tax_rate) * 0.6 + reduction * 0.6
    out13_label.configure(text='市ふるさと控除:' + str(out13))
    out14 = (in10 - 2000) * tax_rate * 0.4 + (in10 - 2000) * 0.04 + (in10 - 2000) * (0.9 - tax_rate) * 0.4 + reduction * 0.4
    out14_label.configure(text='県ふるさと控除:' + str(out14))
    out15 = math.floor(out9/100) * 100 + math.floor(out10/100) * 100 + out11 + out12 - (out13 + out14)
    out15_label.configure(text='住民税:' + str(out15))
    

root = tkinter.Tk()
root.title('所得税・住民税計算機')
root.geometry('500x700')

in1_label = ttk.Label(root, text='年収:')
in1_label.grid(column=0, row=0, padx=10, pady=5)
in1_box = ttk.Entry(root)
in1_box.grid(column=1, row=0, pady=5)

in2_label = ttk.Label(root, text='社会保険料:')
in2_label.grid(column=0, row=2, pady=5)
in2_box = ttk.Entry(root)
in2_box.grid(column=1, row=2, pady=5)

in3_label = ttk.Label(root, text='新生命保険料:')
in3_label.grid(column=0, row=3, pady=5)
in3_box = ttk.Entry(root)
in3_box.grid(column=1, row=3, pady=5)

in4_label = ttk.Label(root, text='旧生命保険料:')
in4_label.grid(column=0, row=4, pady=5)
in4_box = ttk.Entry(root)
in4_box.grid(column=1, row=4, pady=5)

in5_label = ttk.Label(root, text='介護医療保険料:')
in5_label.grid(column=0, row=5, pady=5)
in5_box = ttk.Entry(root)
in5_box.grid(column=1, row=5, pady=5)

in6_label = ttk.Label(root, text='基礎控除:')
in6_label.grid(column=0, row=9, pady=5)
in6_box = ttk.Entry(root)
in6_box.grid(column=1, row=9, pady=5)

in7_label = ttk.Label(root, text='扶養控除:')
in7_label.grid(column=0, row=10, pady=5)
in7_box = ttk.Entry(root)
in7_box.grid(column=1, row=10, pady=5)

in8_label = ttk.Label(root, text='基礎控除:')
in8_label.grid(column=3, row=9, pady=5)
in8_box = ttk.Entry(root)
in8_box.grid(column=4, row=9, pady=5)

in9_label = ttk.Label(root, text='扶養控除:')
in9_label.grid(column=3, row=10, pady=5)
in9_box = ttk.Entry(root)
in9_box.grid(column=4, row=10, pady=5)

in10_label = ttk.Label(root, text='ふるさと納税額:')
in10_label.grid(column=3, row=2, pady=5)
in10_box = ttk.Entry(root)
in10_box.grid(column=4, row=2, pady=5)

in11_label = ttk.Label(root, text='地震保険料:')
in11_label.grid(column=0, row=7, pady=5)
in11_box = ttk.Entry(root)
in11_box.grid(column=1, row=7, pady=5)

calc_btn = ttk.Button(root, text='計算', command=tax_calc)
calc_btn.grid(column=0, row=20, pady=5)

out1_label = ttk.Label(root, text='給与所得')
out1_label.grid(column=1, row=1, pady=5)
out2_label = ttk.Label(root, text='生命保険料控除')
out2_label.grid(column=1, row=6, pady=5)
out3_label = ttk.Label(root, text='所得控除の合計')
out3_label.grid(column=1, row=11, pady=5)
out4_label = ttk.Label(root, text='課税所得')
out4_label.grid(column=1, row=12, pady=5)
out5_label = ttk.Label(root, text='所得税')
out5_label.grid(column=1, row=13, pady=5)
out6_label = ttk.Label(root, text='生命保険料控除')
out6_label.grid(column=3, row=6, pady=5)
out7_label = ttk.Label(root, text='所得控除の合計')
out7_label.grid(column=3, row=11, pady=5)
out8_label = ttk.Label(root, text='課税所得')
out8_label.grid(column=3, row=12, pady=5)
out9_label = ttk.Label(root, text='市民税額')
out9_label.grid(column=3, row=13, pady=5)
out10_label = ttk.Label(root, text='県民税額')
out10_label.grid(column=3, row=14, pady=5)
out11_label = ttk.Label(root, text='市均等割')
out11_label.grid(column=3, row=15, pady=5)
out12_label = ttk.Label(root, text='県均等割')
out12_label.grid(column=3, row=16, pady=5)
out13_label = ttk.Label(root, text='市ふるさと控除')
out13_label.grid(column=3, row=17, pady=5)
out14_label = ttk.Label(root, text='県ふるさと控除')
out14_label.grid(column=3, row=18, pady=5)
out15_label = ttk.Label(root, text='住民税')
out15_label.grid(column=3, row=19, pady=5)
out16_label = ttk.Label(root, text='地震保険料控除')
out16_label.grid(column=1, row=8, pady=5)
out17_label = ttk.Label(root, text='地震保険料控除')
out17_label.grid(column=3, row=8, pady=5)

root.mainloop()



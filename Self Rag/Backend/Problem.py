def oklch(fun):        
    list1 = list(fun)
    first_in = ''.join(list1[0:6])
    last_in = list1[-1]
    str_paramitter = ''.join(list1[6:-1])
    split_ch = ''
    L = 0
    C = 0
    H = 0
    A = 0
    result = []

    if first_in == 'oklch(' and last_in == ')' and first_in.islower():
  
        temp_l = str_paramitter.split()
        if len(temp_l) == 5 or len(temp_l) == 3:
      
            
            t_l = temp_l[0]
            t_c = temp_l[1]
            t_h = temp_l[2]
            if len(temp_l) == 5:
                t_a = temp_l[4]

            if t_l[-1] == '%':
                L = float(t_l[0:-1])
                if 0<L<100:
                    result.append(True)
                else:
                    result.append(False)

            elif t_l == 'none':
                result.append(True)

            elif t_l.replace('.','').isdigit() == True:
                L = float(t_l)
                if 0<L<1:
                    result.append(True)
                else:
                    result.append(False)
            else:
                result.append(False)


            
            if t_c[-1] == '%':
               
                C = float(t_c[0:-1])
                if 0<C<100:
                    result.append(True)
                else:
                    result.append(False)

            elif t_c == 'none':
                result.append(True)
                
            elif t_c.replace('.','').isdigit() == True:                
                C = float(t_c)
                if 0<C<0.4:
                    result.append(True)
                else:
                    result.append(False)
            else:
                            
                result.append(False)



            if t_h == 'none':
                result.append(True)

            elif t_h.replace('.','').isdigit() == True:
                H = float(t_h)
                if 0<H<360:
                    result.append(True)
                else:
                    result.append(False)
            else:
                result.append(False)


            if len(temp_l) == 5:
                if t_a[-1] == '%':
                
                    A = float(t_a[0:-1])
                    if 0<A<100:
                        result.append(True)
                    else:
                        result.append(False)

                
                elif t_c.replace('.','').isdigit() == True:
                    print()
                    A = float(t_a)
                    if 0<A<1:
                        result.append(True)
                    else:
                        result.append(False)
                else:
                    result.append(False)

        else:
            result.append(False)

    else:
        result.append(False)


    f_result = False
    r_count = 0
    print(result)
    for r in result:
        if r == False:
            r_count = r_count + 1
    
    if r_count == 0:
        f_result = True
    else:
        f_result = False



            

    return f_result
    

out = oklch("OKLCH(0.5 0.2 180)")

print(out)
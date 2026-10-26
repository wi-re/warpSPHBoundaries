"""Reuse frozen full-output accuracy/benchmark gates for lifetime variants."""
import fluxbridge_shared_jk_shortmath_study as study
import fluxbridge_shared_jk_lifetime as impl

if __name__=='__main__':
    study.impl=impl
    study.main()
